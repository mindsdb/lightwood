import numpy as np
import pandas as pd
from typing import Dict, List, Union

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.model.lightgbm import LightGBM
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class LightGBMArray(LightGBM):
    """LightGBM-based model, intended for usage in time series tasks. It is trained for t+1 using all
    available data, then partially fit using its own output for longer time horizons."""
    n_ts_predictions: int
    target: str
    supports_proba: bool

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str],
                 n_ts_predictions: int, fit_on_dev: bool, use_optuna: bool = True):
        super().__init__(stop_after, target, dtype_dict, input_cols, fit_on_dev, use_optuna)
        self.n_ts_predictions = n_ts_predictions  # for time series tasks, how long is the forecast horizon
        self.stable = True

    # fit should be different: do the displacement and train on all timesteps using the lgbm output as well
    # partial fit? disable
    # call: same, displace
    # should have a method for displacing

    def _displace_ds(self, data: EncodedDs, predictions: pd.DataFrame):
        # we have predictions that will be incorporated autoregressively
        return data

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting LGBM models for array prediction')

        # fit as normal
        log.info('Fitting T+1')
        super().fit(ds_arr)

        # fit t+n using t+1 predictions
        for timestep in range(1, self.n_ts_predictions):
            new_ds_arr = []
            for ds in ds_arr:
                predictions = self(ds)
                new_ds = self._displace_ds(ds, predictions)
                new_ds_arr.append(new_ds)

            log.info(f'Fitting T+{timestep}')
            super().fit(new_ds_arr)



    def partial_fit(self, train_data: List[EncodedDs], dev_data: List[EncodedDs]) -> None:
        log.info('Updating array of LGBM models...')
        self.model.partial_fit(train_data, dev_data)

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs], predict_proba: bool = False) -> pd.DataFrame:
        if predict_proba:
            log.warning('This model does not output probability estimates')

        ds = ConcatedEncodedDs([ds]) if isinstance(ds, EncodedDs) else ds
        length = sum(ds.encoded_ds_lenghts)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.n_ts_predictions)])

        for timestep in range(self.n_ts_predictions):
            preds = self.model(ds)
            ydf[f'prediction_{timestep}'] = preds
            self._offset_prev_values(ds, preds)

        ydf['prediction'] = ydf.values.tolist()
        return ydf[['prediction']]

    def _offset_prev_values(self, ds: ConcatedEncodedDs, preds: pd.DataFrame):
        for (_, pred), (i, row) in zip(preds.iterrows(), ds.data_frame.iterrows()):
            delta = (row['Date'][-1] - row['Date'][-2])
            row['Date'] = row['Date'][1:] + [row['Date'][-1] + delta]
            row[f'__mdb_ts_previous_{self.target}'] = row[f'__mdb_ts_previous_{self.target}'][1:] + [pred['prediction']]
            ds.data_frame.loc[i] = row
        return ds