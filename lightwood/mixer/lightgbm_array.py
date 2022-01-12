from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import lightgbm

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.encoder import BaseEncoder
from lightwood.mixer.lightgbm import LightGBM
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class LightGBMArray(LightGBM):
    """
    LightGBM-based model, intended for usage in time series tasks.  It is trained for t+1 using all
    available data, then partially fit using its own output for longer time horizons.
    """
    n_ts_predictions: int
    target: str
    supports_proba: bool

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str],
                 n_ts_predictions: int, ts_analysis: Dict, fit_on_dev: bool, target_encoder: BaseEncoder,
                 use_optuna: bool = True, exhaustive_fitting: bool = False):

        super().__init__(stop_after, target, dtype_dict, input_cols, fit_on_dev, use_optuna, target_encoder)
        self.float_dtypes = [dtype.float, dtype.quantity, dtype.array, dtype.tsarray]
        self.num_dtypes = [dtype.integer] + self.float_dtypes
        self.exhaustive_fitting = exhaustive_fitting  # fit t+n using t+(n-1) predictions as historical context
        self.n_ts_predictions = n_ts_predictions  # for time series tasks, how long is the forecast horizon
        self.ts_analysis = ts_analysis
        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting LGBM mixer for array prediction')
        log.info(f'Exhaustive mode: {"enabled" if self.exhaustive_fitting else "disabled"}')
        super().fit(train_data, dev_data)  # fit T+1 as normal

        if self.exhaustive_fitting:
            use_optuna = self.use_optuna
            self.use_optuna = False

            temp_ds = [deepcopy(ds) for ds in (train_data, dev_data)]  # avoids mutating original ds
            for timestep in range(1, self.n_ts_predictions):
                new_ds = []
                for ds in temp_ds:
                    predictions = super().__call__(ds)
                    new_ds.append(self._displace_ds(ds, predictions, timestep))

                log.info(f'Fitting T+{timestep + 1}')
                super().fit(*new_ds)
                temp_ds = new_ds

            self.use_optuna = use_optuna

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        This mixer uses ALL data to re-fit, instead of the normal version that uses
        new data as training data, and old data as validation data (which would be
        incorrect in this case).
        """
        iterations = max(1, int(self.num_iterations))
        data = {'retrain': {'ds': ConcatedEncodedDs([train_data, dev_data]), 'data': None, 'label_data': {}}}
        output_dtype = self.dtype_dict[self.target]
        data = self._to_dataset(data, output_dtype)

        train_dataset = lightgbm.Dataset(data['retrain']['data'],
                                         label=data['retrain']['label_data'],
                                         weight=data['retrain']['weights'])

        log.info(f'Updating lightgbm model with {iterations} iterations')
        self.params['num_iterations'] = int(iterations)
        self.model = lightgbm.train(
            self.params, train_dataset,
            valid_sets=[train_dataset],
            valid_names=['retrain'],
            verbose_eval=False,
            init_model=self.model)
        log.info(f'Model now has a total of {self.model.num_trees()} weak estimators')

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            log.warning('This model does not output probability estimates')

        # force list of EncodedDs, as ConcatedEncodedDs does not support modifying its dataframe
        ds_arr = ds.encoded_ds_arr if isinstance(ds, ConcatedEncodedDs) else [ds]
        temp_ds = [deepcopy(ds) for ds in ds_arr]
        length = sum([len(d) for d in temp_ds])

        ydf = pd.DataFrame(0,
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.n_ts_predictions)])

        for timestep in range(self.n_ts_predictions):
            all_predictions = []
            for i, ds in enumerate(temp_ds):
                predictions = super().__call__(ds, args)
                all_predictions.append(predictions)
                if timestep + 1 < self.n_ts_predictions:
                    temp_ds[i] = self._displace_ds(ds, predictions, timestep + 1)

            all_predictions = pd.concat(all_predictions).reset_index(drop=True)
            ydf[f'prediction_{timestep}'] = all_predictions

        ydf['prediction'] = ydf.values.tolist()
        return ydf[['prediction']]

    def _displace_ds(self, data: EncodedDs, predictions: pd.DataFrame, timestep: int):
        """Moves all array columns one timestep ahead, inserting the LGBM model predictions as historical context"""
        predictions.index = data.data_frame.index
        data.data_frame['__mdb_predictions'] = predictions

        # add prediction to history and displace order by column
        for idx, row in data.data_frame.iterrows():
            histcol = f'__mdb_ts_previous_{self.target}'
            if histcol in data.data_frame.columns:
                data.data_frame.at[idx, histcol] = row.get(histcol)[1:] + [row.get('__mdb_predictions')]

            for col in self.ts_analysis['tss'].order_by:
                if col in data.data_frame.columns:
                    deltas = self.ts_analysis['deltas']
                    group = frozenset(row[self.ts_analysis['tss'].group_by]) \
                        if self.ts_analysis['tss'].group_by \
                        else '__default'  # used w/novel group

                    delta = deltas.get(group, deltas['__default'])[col]
                    data.data_frame.at[idx, col] = row.get(col)[1:] + [row.get(col)[-1] + delta]

        # change target if training
        if f'{self.target}_timestep_{timestep}' in data.data_frame.columns:
            data.data_frame[self.target] = data.data_frame[f'{self.target}_timestep_{timestep}']

        data.data_frame.pop('__mdb_predictions')  # drop temporal column
        return data
