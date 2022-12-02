import time
import math
import torch
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Union, Optional
from optuna import trial as trial_module
from sklearn import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from type_infer.dtype import dtype
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
        :param fit_on_dev: whether to fit on the dev dataset.
        :param use_optuna: whether to activate the automated hyperparameter search (optuna-based). Note that setting this flag to `True` does not guarantee the search will run, rather, the speed criteria will be checked first (i.e., if a single iteration is too slow with respect to the time budget, the search will not take place). 
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
        self.cv = 3
        self.map = {}

        self.cls_dtypes = [dtype.categorical, dtype.binary, dtype.cat_tsarray]
        self.float_dtypes = [dtype.float, dtype.quantity, dtype.num_tsarray]
        self.num_dtypes = [dtype.integer] + self.float_dtypes
        self.supports_proba = dtype_dict[target] in self.cls_dtypes
        self.is_classifier = self.supports_proba

        self.stable = True

    def _multi_logloss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15):
        # ('evaluate model effects' not  use this function)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        score = np.mean([-math.log(y_pred[i][self.map[y]]) for i, y in enumerate(y_true)])

        return score

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

        # concat the data if fit on dev
        if self.fit_on_dev:
            train_data = ConcatedEncodedDs([train_data, dev_data])

        # initialize the model
        init_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'max_features': 1.,
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 0
        }

        if self.is_classifier:
            X = train_data.get_encoded_data(include_target=False)
            Y = train_data.get_column_original_data(self.target)

            self.model = RandomForestClassifier(**init_params)

            self.model.fit(X, Y)  # sample_weight

            self.map = {cat: idx for idx, cat in enumerate(self.model.classes_)}  # for multi_logloss
        else:
            X = train_data.get_encoded_data(include_target=False)
            Y = train_data.get_encoded_column_data(self.target)

            self.model = RandomForestRegressor(**init_params)

            self.model.fit(X, Y)  # sample_weight

        # optimize params
        metric, predict_method = (self._multi_logloss, 'predict_proba') if self.is_classifier \
            else (mean_squared_error, 'predict')

        def objective(trial: trial_module.Trial):
            criterion = trial.suggest_categorical("criterion",
                                                  ["gini", "entropy"]) if self.is_classifier else 'squared_error'

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 2, 512),
                'max_depth': trial.suggest_int('max_depth', 2, 15),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
                'max_features': trial.suggest_float("max_features", 0.1, 1),
                'criterion': criterion,
            }

            self.model.set_params(**params)

            y_pred = cross_val_predict(self.model, X, Y, cv=self.cv, method=predict_method)
            cv = check_cv(self.cv, Y, classifier=self.is_classifier)  #
            score = np.mean([metric(np.array(Y)[val_idx], y_pred[val_idx]) for _, val_idx in cv.split(X, Y)])

            return score

        elapsed = time.time() - started
        num_trials = max(min(int(self.stop_after / elapsed) - 2, self.num_trials), 0)
        if self.use_optuna:
            log.info(f'The number of trials (Optuna) is {num_trials}.')

        if self.use_optuna and num_trials > 0:
            init_score = metric(Y, getattr(self.model, predict_method)(X))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=num_trials)

            opt_model = clone(self.model)
            opt_model.set_params(**study.best_params)
            opt_model.fit(X, Y)
            optuna_score = metric(Y, getattr(opt_model, predict_method)(X))
            log.info(f'init_score: {init_score}, optuna_score: {optuna_score}')

            if init_score <= optuna_score:
                self.model.set_params(**init_params)
            else:
                self.model = opt_model
                log.info(f'RandomForest parameters of the best trial: {study.best_params}')

        # evaluate model effects
        if self.fit_on_dev:
            log.info(f'RandomForest based correlation of (train data): {self.model.score(X, Y)}')
            X = dev_data.get_encoded_data(include_target=False)
            if self.is_classifier:
                Y = dev_data.get_column_original_data(self.target)
            else:
                Y = dev_data.get_encoded_column_data(self.target)
            log.info(f'RandomForest based correlation of (dev data): {self.model.score(X, Y)}')
        else:
            log.info(f'RandomForest based correlation of: {self.model.score(X, Y)}')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs, args: Optional[dict] = None) -> None:
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
        data = ds.get_encoded_data(include_target=False)

        if self.is_classifier:
            predictions = self.model.predict_proba(data)
            decoded_predictions = self.model.classes_.take(np.argmax(predictions, axis=1), axis=0)
        else:
            predictions = self.model.predict(data)
            if predictions.ndim == 1:
                decoded_predictions = predictions
            else:
                decoded_predictions = self.target_encoder.decode(torch.Tensor(predictions))

        if self.positive_domain:
            decoded_predictions = [max(0, p) for p in decoded_predictions]

        ydf = pd.DataFrame({'prediction': decoded_predictions})

        if args.predict_proba and self.supports_proba:
            for idx, label in enumerate(self.model.classes_):
                ydf[f'__mdb_proba_{label}'] = predictions[:, idx]

        return ydf
