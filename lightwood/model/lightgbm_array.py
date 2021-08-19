import numpy as np
import pandas as pd
from typing import Dict, List, Union

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.model.lightgbm import LightGBM
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class LightGBMArray(BaseModel):
    """LightGBM-based model, intended for usage in time series tasks."""
    models: List[LightGBM]
    n_ts_predictions: int
    submodel_stop_after: float
    target: str
    supports_proba: bool

    def __init__(
            self, stop_after: int, target: str, dtype_dict: Dict[str, str],
            input_cols: List[str],
            n_ts_predictions: int, fit_on_dev: bool):
        super().__init__(stop_after)
        self.submodel_stop_after = stop_after / n_ts_predictions
        self.target = target
        dtype_dict[target] = dtype.float
        self.model = LightGBM(self.submodel_stop_after, target, dtype_dict, input_cols, fit_on_dev, use_optuna=False)
        self.n_ts_predictions = n_ts_predictions  # for time series tasks, how long is the forecast horizon
        self.supports_proba = False
        self.stable = True

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting LGBM models for array prediction')
        self.model.fit(ds_arr)

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