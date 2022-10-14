import time
import torch
import numpy as np
import pandas as pd
import optuna
from optuna import trial as trial_module
from sklearn.model_selection import cross_val_score
from typing import Dict, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.mixer.base import BaseMixer
from lightwood.api.types import PredictionArguments


class RandomForest(BaseMixer):
    model: Union[RandomForestClassifier, RandomForestRegressor]
    dtype_dict: dict
    target: str
    fit_on_dev: bool
    use_optuna: bool
    supports_proba: bool

    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            fit_on_dev: bool,
            use_optuna: bool,
            target_encoder: BaseEncoder
    ):
        """
        The `RandomForest` mixer supports both regression and classification tasks. 
        It inherits from sklearn.ensemble.RandomForestRegressor and sklearn.ensemble.RandomForestClassifier.
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        
        :param stop_after: time budget in seconds.
        :param target: name of the target column that the mixer will learn to predict.
        :param dtype_dict: dictionary with dtypes of all columns in the data.
        :param fit_on_dev: whether to perform a `partial_fit()` at the end of `fit()` using the `dev` data split.
        :param use_optuna: whether to activate the automated hyperparameter search. 
        """  # noqa
        super().__init__(stop_after)
        self.target = target
        self.dtype_dict = dtype_dict
        self.fit_on_dev = fit_on_dev
        self.use_optuna = use_optuna
        self.target_encoder = target_encoder

        self.model = None
        self.positive_domain = False
        self.num_trials = 20

        self.cls_dtypes = [dtype.categorical, dtype.binary, dtype.cat_tsarray]
        self.float_dtypes = [dtype.float, dtype.quantity, dtype.num_tsarray]
        self.num_dtypes = [dtype.integer] + self.float_dtypes
        self.supports_proba = dtype_dict[target] in self.cls_dtypes

        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the RandomForest model.

        :param train_data: encoded features for training dataset
        :param dev_data: encoded features for dev dataset
        """
        started = time.time()
        log.info('Started fitting RandomForest model')

        output_dtype = self.dtype_dict[self.target]

        if output_dtype not in self.cls_dtypes + self.num_dtypes:
            log.error(f'RandomForest mixer not supported for type: {output_dtype}')
            raise Exception(f'RandomForest mixer not supported for type: {output_dtype}')

        if self.fit_on_dev:
            train_data = ConcatedEncodedDs([train_data, dev_data])

        if output_dtype in self.num_dtypes:
            X = train_data.get_encoded_data(include_target=False)
            try:
                Y = train_data.get_encoded_column_data(self.target)
            except Exception as e:
                log.warning(e)
                Y = train_data.get_column_original_data(self.target)  # ts: to be fixed

            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                max_features=1.,
                bootstrap=True,
                n_jobs=-1,
                random_state=0
            )

            self.model.fit(X, Y)  # sample_weight

        elif output_dtype in self.cls_dtypes:
            X = train_data.get_encoded_data(include_target=False)
            Y = train_data.get_column_original_data(self.target)

            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                max_features=1.,
                bootstrap=True,
                n_jobs=-1,
                random_state=0
            )

            self.model.fit(X, Y)  # sample_weight

        # need to be improved
        elapsed = time.time() - started
        num_trials = max(min(int(self.stop_after / elapsed) - 1, self.num_trials), 0)
        if self.use_optuna:
            log.info(f'The number of trials (Optuna) is {num_trials}.')

        direction, metric = ('maximize', 'r2') if output_dtype in self.num_dtypes else ('maximize', 'neg_log_loss')

        def objective(trial: trial_module.Trial):
            criterion = 'squared_error' if output_dtype in self.num_dtypes \
                else trial.suggest_categorical("criterion", ["gini", "entropy"])

            params = {
                'n_estimators': trial.suggest_int('num_estimators', 2, 512),
                'max_depth': trial.suggest_int('max_depth', 2, 15),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 100),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 100),
                'max_features': trial.suggest_float("max_features", 0.01, 1),
                'criterion': criterion,
            }

            self.model.set_params(**params)

            return cross_val_score(self.model, X, Y, cv=3, n_jobs=-1, scoring=metric).mean()

        if self.use_optuna and num_trials > 0:
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=num_trials)
            # study.optimize(objective, n_trials=self.num_trials)
            # to be fixed
            # print(study.trials_dataframe().tail())
            # log.info(f'RandomForest parameters of the best trial: {study.best_params}')
            log.info(f'RandomForest n_estimators: {self.model.n_estimators},  max_depth: {self.model.max_depth}')

        log.info(f'RandomForest based correlation of: {self.model.score(X, Y)}')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        The RandomForest mixer does not support updates. If the model does not exist, a new one will be created and fitted. 
        
        :param train_data: encoded features for (new) training dataset
        :param dev_data: encoded features for (new) dev dataset
        """  # noqa
        if self.model is None:
            self.fit(train_data, dev_data)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Call a trained RandomForest mixer to output predictions for the target column.

        :param ds: input data with values for all non-target columns.
        :param args: inference-time arguments (e.g. whether to output predicted labels or probabilities).

        :return: dataframe with predictions.
        """
        data = ds.get_encoded_data(include_target=False).tolist()

        if self.dtype_dict[self.target] in self.num_dtypes:
            predictions = self.model.predict(data)
            if predictions.ndim == 1:
                decoded_predictions = predictions
            else:
                decoded_predictions = self.target_encoder.decode(torch.Tensor(predictions))
        else:
            predictions = self.model.predict_proba(data)
            decoded_predictions = self.model.classes_.take(np.argmax(predictions, axis=1), axis=0)

        if self.positive_domain:
            decoded_predictions = [max(0, p) for p in decoded_predictions]

        ydf = pd.DataFrame({'prediction': decoded_predictions})

        if args.predict_proba and hasattr(self.model, 'classes_'):
            for idx, label in enumerate(self.model.classes_):
                ydf[f'__mdb_proba_{label}'] = predictions[:, idx]

        return ydf

