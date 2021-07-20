import sktime
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class SkTime(BaseModel):
    forecaster: str
    n_ts_predictions:  int
    # submodel_stop_after: float
    target: str

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str], n_ts_predictions: int):
        super().__init__(stop_after)
        self.target = target
        dtype_dict[target] = dtype.float
        # self.model = NaiveForecaster(strategy="last")
        self.model = ThetaForecaster()
        self.n_ts_predictions = n_ts_predictions
        self.forecasting_horizon = np.arange(1, self.n_ts_predictions)

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting sktime forecaster for array prediction')

        for fold in range(len(ds_arr)):
            # ds_arr[fold].data_frame[self.target] = ds_arr[fold].data_frame[f'{self.target}_timestep_{timestep}']
            self.model.fit(ds_arr[fold].data_frame[self.target])

            # index = [row[-1][0][-1] for idx, row in ds_arr[fold].data_frame[['T']].iterrows()]
            # d = pd.Series(ds_arr[fold].data_frame[self.target].values, index=pd.Int64Index(index))
            # d = d.sort_index(ascending=True)
            # self.model.fit(d)

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs]) -> pd.DataFrame:
        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=['prediction'],
                           dtype=object)

        print(ydf)

        # ydf['prediction'] = self.model.predict(ds.data_frame[self.target].index).tolist()

        for new_idx, (_, _) in enumerate(ds.data_frame.iterrows()):
            ydf['prediction'].iloc[new_idx] = self.model.predict(np.arange(self.n_ts_predictions)).tolist()
            # ydf[f'prediction_{timestep}'] = self.models[timestep](ds)

        # ydf['prediction'] = ydf.values.tolist()

        print(ydf[['prediction']])
        print(ydf[['prediction']].shape)

        return ydf[['prediction']]
