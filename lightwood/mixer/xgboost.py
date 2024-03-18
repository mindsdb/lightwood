from typing import Dict, List, Optional, Union
import time

import torch
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from type_infer.dtype import dtype

from lightwood.helpers.log import log
from lightwood.helpers.parallelism import get_nr_procs
from lightwood.mixer.base import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs

optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def check_gpu_support():
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        x, _, y, _ = train_test_split(data['data'], data['target'], test_size=.2)
        bst = xgb.XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic',
                                tree_method='gpu_hist', gpu_id=0)
        bst.fit(x, y)
        return True
    except xgb.core.XGBoostError:
        return False


class XGBoostMixer(BaseMixer):
    model: Union[xgb.XGBClassifier, xgb.XGBRegressor]
    ordinal_encoder: OrdinalEncoder
    label_set: List[str]
    max_bin: int
    device: torch.device
    device_str: str
    num_iterations: int
    use_optuna: bool
    supports_proba: bool

    """
    Gradient boosting mixer with an XGBoost backbone.

    This mixer is a good all-rounder, due to the generally great performance of tree-based ML algorithms for supervised learning tasks with tabular data.
    If you want more information regarding the techniques that set apart XGBoost from other gradient boosters, please refer to their technical paper: "XGBoost: A Scalable Tree Boosting System" (2016).

    We can basically think of this mixer as a wrapper to the XGBoost Python package. To do so, there are a few caveats the user may want to be aware about:
        * If you seek GPU utilization, XGBoost must be compiled from source instead of being installed through `pip`.
        * Integer, float, and quantity `dtype`s are treated as regression tasks with `reg:squarederror` loss. All other supported `dtype`s is casted as a multiclass task with `multi:softmax` loss.
        * A partial fit can be performed with the `dev` data split as part of `fit`, if specified with the `fit_on_dev` argument.

    There are a couple things in the backlog that will hopefully be added soon:
        * An automatic optuna-based hyperparameter search. This procedure triggers when a single iteration of XGBoost is deemed fast enough (given the time budget).
        * Support for "unknown class" as a possible answer for multiclass tasks.
    """  # noqa

    def __init__(
            self, stop_after: float, target: str, dtype_dict: Dict[str, str],
            input_cols: List[str],
            fit_on_dev: bool, use_optuna: bool, target_encoder: BaseEncoder):
        """
        :param stop_after: time budget in seconds.
        :param target: name of the target column that the mixer will learn to predict.
        :param dtype_dict: dictionary with dtypes of all columns in the data.
        :param input_cols: list of column names.
        :param fit_on_dev: whether to perform a `partial_fit()` at the end of `fit()` using the `dev` data split.
        :param use_optuna: whether to activate the automated hyperparameter search (optuna-based). Note that setting this flag to `True` does not guarantee the search will run, rather, the speed criteria will be checked first (i.e., if a single iteration is too slow with respect to the time budget, the search will not take place). 
        """  # noqa
        super().__init__(stop_after)
        self.model = None
        self.ordinal_encoder = None
        self.positive_domain = False
        self.label_set = []
        self.target = target
        self.dtype_dict = dtype_dict
        self.input_cols = input_cols
        self.use_optuna = use_optuna
        self.params = {}
        self.fit_on_dev = fit_on_dev
        self.cls_dtypes = [dtype.categorical, dtype.binary, dtype.cat_tsarray]
        self.float_dtypes = [dtype.float, dtype.quantity, dtype.num_tsarray]
        self.num_dtypes = [dtype.integer] + self.float_dtypes
        self.supports_proba = dtype_dict[target] in self.cls_dtypes
        self.stable = True
        self.target_encoder = target_encoder

        gpu_works = check_gpu_support()
        if not gpu_works:
            self.device = torch.device('cpu')
            self.device_str = 'cpu'
            log.warning('XGBoost running on CPU')
        else:
            self.device = torch.device('cuda')
            self.device_str = 'gpu'

        self.max_bin = 255

    def _to_dataset(self, ds: EncodedDs, output_dtype: str, mode='train'):
        """
        Helper method to wrangle a datasource into the format that the underlying model requires.
        :param ds: EncodedDS that contains the dataframe to transform.
        :param output_dtype:
        :return: modified `data` object that conforms to XGBoost's expected format.
        """  # noqa
        data = None
        for input_col in self.input_cols:
            if data is None:
                data = ds.get_encoded_column_data(input_col).to(self.device)
            else:
                enc_col = ds.get_encoded_column_data(input_col).to(self.device)
                data = torch.cat((data, enc_col.to(self.device)), 1)

        data = data.cpu().numpy()

        if mode in ('train', 'dev'):
            weights = []
            label_data = ds.get_column_original_data(self.target)
            if output_dtype in self.cls_dtypes:
                if mode == 'train':
                    self.ordinal_encoder = OrdinalEncoder()
                    self.label_set = list(set(label_data))
                    self.ordinal_encoder.fit(np.array(list(self.label_set)).reshape(-1, 1))

                filtered_label_data = []
                for x in label_data:
                    if x in self.label_set:
                        filtered_label_data.append(x)

                weight_map = getattr(self.target_encoder, 'target_weights', None)
                if weight_map is not None:
                    weights = [weight_map[x] for x in label_data]

                label_data = self.ordinal_encoder.transform(np.array(filtered_label_data).reshape(-1, 1)).flatten()

            elif output_dtype in self.num_dtypes:
                weight_map = getattr(self.target_encoder, 'target_weights', None)
                if weight_map is not None:
                    target_encoder = ds.encoders[self.target]

                    # get the weights from the numeric target encoder
                    weights = target_encoder.get_weights(label_data)

                if output_dtype in self.float_dtypes:
                    label_data = label_data.astype(float)
                elif output_dtype == dtype.integer:
                    label_data = label_data.clip(-pow(2, 63), pow(2, 63)).astype(int)

            return data, label_data, weights

        else:
            return data

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the XGBoost model.

        :param train_data: encoded features for training dataset
        :param dev_data: encoded features for dev dataset
        """
        started = time.time()

        log.info('Started fitting XGBoost model')
        self.fit_data_len = len(train_data)
        self.positive_domain = getattr(train_data.encoders.get(self.target, None), 'positive_domain', False)
        output_dtype = self.dtype_dict[self.target]

        if output_dtype not in self.cls_dtypes + self.num_dtypes:
            log.error(f'XGBoost mixer not supported for type: {output_dtype}')
            raise Exception(f'XGBoost mixer not supported for type: {output_dtype}')
        else:
            objective = 'reg:squarederror' if output_dtype in self.num_dtypes else 'multi:softprob'
            metric = 'rmse' if output_dtype in self.num_dtypes else 'mlogloss'

        self.params = {
            'objective': objective,
            'eval_metric': metric,
            'n_jobs': get_nr_procs(train_data.data_frame),
            'process_type': 'default',  # normal training
            'verbosity': 0,
            'early_stopping_rounds': 5
            # 'device_type': self.device_str,  # TODO
        }

        # Prepare the data
        train_dataset, train_labels, train_weights = self._to_dataset(train_data, output_dtype, mode='train')
        dev_dataset, dev_labels, dev_weights = self._to_dataset(dev_data, output_dtype, mode='dev')

        if output_dtype not in self.num_dtypes:
            self.all_classes = self.ordinal_encoder.categories_[0]
            self.params['num_class'] = self.all_classes.size
            model_class = xgb.XGBClassifier
        else:
            model_class = xgb.XGBRegressor

        # Determine time per iterations
        start = time.time()
        self.params['n_estimators'] = 1

        with xgb.config_context(verbosity=0):
            self.model = model_class(**self.params)
            if train_weights is not None and dev_weights is not None:
                self.model.fit(train_dataset, train_labels, sample_weight=train_weights,
                               eval_set=[(dev_dataset, dev_labels)],
                               sample_weight_eval_set=[dev_weights])
            else:
                self.model.fit(train_dataset, train_labels,
                               eval_set=[(dev_dataset, dev_labels)])

        end = time.time()
        seconds_for_one_iteration = max(0.1, end - start)

        self.stop_after = max(1, self.stop_after - (time.time() - started))

        # Determine nr of iterations
        log.info(f'A single GBM iteration takes {seconds_for_one_iteration} seconds')
        self.num_iterations = int(self.stop_after * 0.8 / seconds_for_one_iteration)

        # TODO: Turn on grid search if training doesn't take too long using it
        # kwargs = {}
        # if self.use_optuna and self.num_iterations >= 200:
        #     model_generator = None
        #     kwargs['time_budget'] = self.stop_after * 0.4
        #     self.num_iterations = int(self.num_iterations / 2)
        #     kwargs['optuna_seed'] = 0
        # else:
        #     model_generator = None

        # Train the models
        log.info(
            f'Training XGBoost with {self.num_iterations} iterations given {self.stop_after} seconds constraint')  # noqa
        if self.num_iterations < 1:
            self.num_iterations = 1
        self.params['n_estimators'] = int(self.num_iterations)

        # TODO: reinstance these on each run! keep or use via eval_set in fit()?
        # self.params['callbacks'] = [(train_dataset, 'train'), (dev_dataset, 'eval')]

        with xgb.config_context(verbosity=0):
            self.model = model_class(**self.params)
            if train_weights is not None and dev_weights is not None:
                self.model.fit(train_dataset, train_labels, sample_weight=train_weights,
                               eval_set=[(dev_dataset, dev_labels)],
                               sample_weight_eval_set=[dev_weights])
            else:
                self.model.fit(train_dataset, train_labels,
                               eval_set=[(dev_dataset, dev_labels)])

        if self.fit_on_dev:
            self.partial_fit(dev_data, train_data)

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs, args: Optional[dict] = None) -> None:
        # TODO: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        # To resume training from a previous checkpoint, explicitly pass xgb_model argument.
        log.info('XGBoost mixer does not have a `partial_fit` implementation')

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Call a trained XGBoost mixer to output predictions for the target column.

        :param ds: input data with values for all non-target columns.
        :param args: inference-time arguments (e.g. whether to output predicted labels or probabilities).

        :return: dataframe with predictions.
        """  # noqa
        output_dtype = self.dtype_dict[self.target]
        xgbdata = self._to_dataset(ds, self.dtype_dict[self.target], mode='predict')

        with xgb.config_context(verbosity=0):
            if output_dtype in self.num_dtypes:
                raw_predictions = self.model.predict(xgbdata)
            else:
                raw_predictions = self.model.predict_proba(xgbdata)

        if self.ordinal_encoder is not None:
            decoded_predictions = self.ordinal_encoder.inverse_transform(
                np.argmax(raw_predictions, axis=1).reshape(-1, 1)).flatten()
        else:
            decoded_predictions = raw_predictions

        if self.positive_domain:
            decoded_predictions = [max(0, p) for p in decoded_predictions]

        ydf = pd.DataFrame({'prediction': decoded_predictions})

        if args.predict_proba and self.ordinal_encoder is not None:
            for idx, label in enumerate(self.ordinal_encoder.categories_[0].tolist()):
                ydf[f'__mdb_proba_{label}'] = raw_predictions[:, idx]

        return ydf
