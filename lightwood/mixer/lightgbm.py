import time
from typing import Dict, List, Set
import torch
import optuna
import lightgbm
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import optuna.integration.lightgbm as optuna_lightgbm
from lightwood.api import dtype
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.helpers.device import get_devices
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs


optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def check_gpu_support():
    try:
        data = np.random.rand(50, 2)
        label = np.random.randint(2, size=50)
        train_data = lightgbm.Dataset(data, label=label)
        params = {'num_iterations': 1, 'device': 'gpu'}
        lightgbm.train(params, train_set=train_data)
        device, nr_devices = get_devices()
        if nr_devices > 0 and str(device) != 'cpu':
            return True
        else:
            return False
    except Exception:
        return False


class LightGBM(BaseMixer):
    model: lightgbm.LGBMModel
    ordinal_encoder: OrdinalEncoder
    label_set: Set[str]
    max_bin: int
    device: torch.device
    device_str: str
    num_iterations: int
    use_optuna: bool
    supports_proba: bool

    """
    Gradient boosting mixer with a LightGBM backbone.
    
    This mixer is a good all-rounder, due to the generally great performance of tree-based ML algorithms for supervised learning tasks with tabular data.
    If you want more information regarding the techniques that set apart LightGBM from other gradient boosters, please refer to their technical paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017).
    
    We can basically think of this mixer as a wrapper to the LightGBM interface. To do so, there are a few caveats the user may want to be aware about:
        * If you seek GPU utilization, LightGBM must be compiled from source instead of being installed through `pip`.
        * Integer, float, and quantity `dtype`s are treated as regression tasks with `L2` loss. All other supported `dtype`s is casted as a multiclass task with `multi_logloss` loss.
        * It has an automatic optuna-based hyperparameter search. This procedure triggers when a single iteration of LightGBM is deemed fast enough (given the time budget).
        * A partial fit can be performed with the `dev` data split as part of `fit`, if specified with the `fit_on_dev` argument.
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
        self.label_set = set()
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

        # GPU Only available via --install-option=--gpu with opencl-dev and libboost dev (a bunch of them) installed, so let's turn this off for now and we can put it behind some flag later # noqa
        gpu_works = check_gpu_support()
        if not gpu_works:
            self.device = torch.device('cpu')
            self.device_str = 'cpu'
            log.warning('LightGBM running on CPU, this somewhat slower than the GPU version, consider using a GPU instead') # noqa
        else:
            self.device = torch.device('cuda')
            self.device_str = 'gpu'

        self.max_bin = 255

    def _to_dataset(self, data: Dict[str, Dict], output_dtype: str):
        """
        Helper method to wrangle data into the format that the underlying model requires.

        :param data: Includes train and dev data datasources.
        :param output_dtype
        :return: modified `data` object that conforms to LightGBM's expected format.
        """
        weight_map = getattr(self.target_encoder, 'target_weights', None)

        for subset_name in data.keys():
            for input_col in self.input_cols:
                if data[subset_name]['data'] is None:
                    data[subset_name]['data'] = data[subset_name]['ds'].get_encoded_column_data(
                        input_col).to(self.device)
                else:
                    enc_col = data[subset_name]['ds'].get_encoded_column_data(input_col)
                    data[subset_name]['data'] = torch.cat((data[subset_name]['data'], enc_col.to(self.device)), 1)

            data[subset_name]['data'] = data[subset_name]['data'].cpu().numpy()

            label_data = data[subset_name]['ds'].get_column_original_data(self.target)

            data[subset_name]['weights'] = None
            if output_dtype in self.cls_dtypes:
                if subset_name == 'train':
                    self.ordinal_encoder = OrdinalEncoder()
                    self.label_set = set(label_data)
                    self.label_set.add('__mdb_unknown_cat')
                    self.ordinal_encoder.fit(np.array(list(self.label_set)).reshape(-1, 1))

                label_data = [x if x in self.label_set else '__mdb_unknown_cat' for x in label_data]
                if weight_map is not None:
                    data[subset_name]['weights'] = [weight_map[x] for x in label_data]
                label_data = self.ordinal_encoder.transform(np.array(label_data).reshape(-1, 1)).flatten()
            elif output_dtype == dtype.integer:
                label_data = label_data.clip(-pow(2, 63), pow(2, 63)).astype(int)
            elif output_dtype in self.float_dtypes:
                label_data = label_data.astype(float)

            data[subset_name]['label_data'] = label_data

        return data

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the LightGBM model.

        :param train_data: encoded features for training dataset
        :param dev_data: encoded features for dev dataset
        """
        started = time.time()

        log.info('Started fitting LGBM model')
        data = {
            'train': {'ds': train_data, 'data': None, 'label_data': {}},
            'dev': {'ds': dev_data, 'data': None, 'label_data': {}}
        }
        self.fit_data_len = len(data['train']['ds'])
        self.positive_domain = getattr(train_data.encoders.get(self.target, None), 'positive_domain', False)

        output_dtype = self.dtype_dict[self.target]
        data = self._to_dataset(data, output_dtype)

        if output_dtype not in self.cls_dtypes + self.num_dtypes:
            log.error(f'Lightgbm mixer not supported for type: {output_dtype}')
            raise Exception(f'Lightgbm mixer not supported for type: {output_dtype}')
        else:
            objective = 'regression' if output_dtype in self.num_dtypes else 'multiclass'
            metric = 'l2' if output_dtype in self.num_dtypes else 'multi_logloss'

        self.params = {
            'objective': objective,
            'metric': metric,
            'verbose': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'force_row_wise': True,
            'device_type': self.device_str,
        }

        if objective == 'multiclass':
            self.all_classes = self.ordinal_encoder.categories_[0]
            self.params['num_class'] = self.all_classes.size
        elif output_dtype == dtype.num_tsarray:
            self.params['linear_tree'] = True
        if self.device_str == 'gpu':
            self.params['gpu_use_dp'] = True

        # Determine time per iterations
        start = time.time()
        self.params['num_iterations'] = 1
        '''
        Why construst a dataset here instead of using the training dataset?

        Because it guards against the following crash:

        WARNING:lightwood-1613058:Exception: Cannot change feature_pre_filter after constructed Dataset handle. when training mixer: <lightwood.mixer.lightgbm.LightGBM object at 0x7f39148eae20>
        feature_fraction, val_score: inf:   0%|                                 | 0/7 [00:00<?, ?it/s]
        free(): double free detected in tcache 2

        Only happens sometimes and I can find no pattern as to when, happens for multiple input and target types.

        Why does the following crash happen and what does it mean? No idea, closest relationships I can find is /w optuna modifying parameters after the dataset is create: https://github.com/microsoft/LightGBM/issues/4019 | But why this would apply here makes no sense. Could have to do with the `train` process of lightgbm itself setting a "set only once" property on a dataset when it starts. Dunno, if you find out replace this comment with the real reason.
        ''' # noqa

        self.model = lightgbm.train(self.params, lightgbm.Dataset(data['train']['data'], label=data['train']
                                    ['label_data'], weight=data['train']['weights']), verbose_eval=False)
        end = time.time()
        seconds_for_one_iteration = max(0.1, end - start)

        self.stop_after = max(1, self.stop_after - (time.time() - started))
        # Determine nr of iterations
        log.info(f'A single GBM iteration takes {seconds_for_one_iteration} seconds')
        self.num_iterations = int(self.stop_after * 0.8 / seconds_for_one_iteration)

        # Turn on grid search if training doesn't take too long using it
        kwargs = {}
        if self.use_optuna and self.num_iterations >= 200:
            model_generator = optuna_lightgbm
            kwargs['time_budget'] = self.stop_after * 0.4
            self.num_iterations = int(self.num_iterations / 2)
            kwargs['optuna_seed'] = 0
        else:
            model_generator = lightgbm

        # Train the models
        log.info(
            f'Training GBM ({model_generator}) with {self.num_iterations} iterations given {self.stop_after} seconds constraint') # noqa
        if self.num_iterations < 1:
            self.num_iterations = 1
        self.params['num_iterations'] = int(self.num_iterations)

        self.params['early_stopping_rounds'] = 5

        # Prepare the data
        train_dataset = lightgbm.Dataset(data['train']['data'], label=data['train']['label_data'],
                                         weight=data['train']['weights'])

        dev_dataset = lightgbm.Dataset(data['dev']['data'], label=data['dev']['label_data'],
                                       weight=data['dev']['weights'])

        self.model = model_generator.train(
            self.params, train_dataset, valid_sets=[dev_dataset, train_dataset],
            valid_names=['dev', 'train'],
            verbose_eval=False, **kwargs)
        self.num_iterations = self.model.best_iteration
        log.info(f'Lightgbm model contains {self.model.num_trees()} weak estimators')

        if self.fit_on_dev:
            self.partial_fit(dev_data, train_data)

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Updates the LightGBM model.

        :param train_data: encoded features for (new) training dataset
        :param dev_data: encoded features for (new) dev dataset
        """
        pct_of_original = len(train_data) / self.fit_data_len
        iterations = max(1, int(self.num_iterations * pct_of_original) / 2)

        data = {'retrain': {'ds': train_data, 'data': None, 'label_data': {}}, 'dev': {
            'ds': dev_data, 'data': None, 'label_data': {}}}

        output_dtype = self.dtype_dict[self.target]
        data = self._to_dataset(data, output_dtype)

        train_dataset = lightgbm.Dataset(data['retrain']['data'], label=data['retrain']['label_data'],
                                         weight=data['retrain']['weights'])
        dev_dataset = lightgbm.Dataset(data['dev']['data'], label=data['dev']['label_data'],
                                       weight=data['dev']['weights'])

        log.info(f'Updating lightgbm model with {iterations} iterations')
        self.params['num_iterations'] = int(iterations)
        self.model = lightgbm.train(
            self.params, train_dataset, valid_sets=[dev_dataset, train_dataset],
            valid_names=['dev', 'retrain'],
            verbose_eval=False, init_model=self.model)
        log.info(f'Model now has a total of {self.model.num_trees()} weak estimators')

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Call a trained LightGBM mixer to output predictions for the target column.

        :param ds: input data with values for all non-target columns.
        :param args: inference-time arguments (e.g. whether to output predicted labels or probabilities).

        :return: dataframe with predictions.
        """
        data = None
        for input_col in self.input_cols:
            if data is None:
                data = ds.get_encoded_column_data(input_col).to(self.device)
            else:
                data = torch.cat((data, ds.get_encoded_column_data(input_col).to(self.device)), 1)

        data = data.cpu().numpy()
        raw_predictions = self.model.predict(data)

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
