import time
import torch
import numpy as np
import pandas as pd
import optuna
from optuna import trial as trial_module
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from typing import Dict, List, Set, Optional, Union
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
    num_iterations: int
    fit_on_dev: bool
    use_optuna: bool
    supports_proba: bool

    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            input_cols: List[str],
            fit_on_dev: bool,
            use_optuna: bool,
            target_encoder: BaseEncoder
    ):
        super().__init__(stop_after)
        self.target = target
        self.dtype_dict = dtype_dict
        self.input_cols = input_cols
        self.fit_on_dev = fit_on_dev
        self.use_optuna = use_optuna
        self.target_encoder = target_encoder

        self.model = None
        self.positive_domain = False
        self.num_iterations = None
        self.params = {}

        self.cls_dtypes = [dtype.categorical, dtype.binary, dtype.cat_tsarray]
        self.float_dtypes = [dtype.float, dtype.quantity, dtype.num_tsarray]
        self.num_dtypes = [dtype.integer] + self.float_dtypes
        self.supports_proba = dtype_dict[target] in self.cls_dtypes

        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting RF model')

        output_dtype = self.dtype_dict[self.target]

        if output_dtype not in self.cls_dtypes + self.num_dtypes:
            log.error(f'RandomForest mixer not supported for type: {output_dtype}')
            raise Exception(f'RandomForest mixer not supported for type: {output_dtype}')

        # =========================================== regression ===========================================
        if output_dtype in self.num_dtypes:
            X = train_data.get_encoded_data(include_target=False)
            Y = train_data.get_encoded_column_data(self.target)

            self.model = RandomForestRegressor(
                n_estimators=50, max_depth=5, max_features=1.,
                bootstrap=True, n_jobs=-1, random_state=0
            )

            self.model.fit(X, Y)  # sample_weight
            log.info(f'RandomForest based correlation of: {self.model.score(X, Y)}')

        # ========================================= classification =========================================
        elif output_dtype in self.cls_dtypes:
            X = train_data.get_encoded_data(include_target=False)
            Y = train_data.get_column_original_data(self.target)

            self.model = RandomForestClassifier(
                            n_estimators=5, max_depth=5, max_features=1.,
                            bootstrap=True, n_jobs=-1, random_state=0)

            self.model.fit(X, Y)  # sample_weight
            log.info(f'RandomForest based accuracy of: {self.model.score(X, Y)}')

        # ========================================= params optimization =========================================
        # need to be improved
        direction, metric = ('maximize', 'r2') if output_dtype in self.num_dtypes else ('maximize', 'neg_log_loss')

        def objective(trial: trial_module.Trial):
            n_estimators = trial.suggest_int('num_estimators', 5, 1000)
            max_depth = trial.suggest_int('max_depth', 2, 15)

            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                # ...
            }

            self.model.set_params(**params)

            return cross_val_score(self.model, X, Y, cv=3, n_jobs=-1, scoring=metric).mean()

        if self.use_optuna:
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=20)
            print(study.trials_dataframe().iloc[-1])

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        # self.model.fit(X, Y)
        pass

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        X = ds.get_encoded_data(include_target=False).tolist()

        Yh = self.model.predict(X)
        print('>>>' * 10, 'Yh')
        print(Yh)

        if self.dtype_dict[self.target] in self.num_dtypes:
            decoded_predictions = self.target_encoder.decode(torch.Tensor(Yh))
        else:
            decoded_predictions = Yh

        if self.positive_domain:
            decoded_predictions = [max(0, p) for p in decoded_predictions]

        ydf = pd.DataFrame({'prediction': decoded_predictions})
        print('>>>' * 10, 'ydf')
        print(ydf)

        # ========================================= something else =========================================

        return ydf

