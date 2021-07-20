import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sktime.forecasting.theta import ThetaForecaster

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.encoder.time_series.helpers.common import get_group_matches
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class SkTime(BaseModel):
    forecaster: str
    n_ts_predictions: int
    target: str

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], n_ts_predictions: int, ts_analysis: Dict):
        super().__init__(stop_after)
        self.target = target
        dtype_dict[target] = dtype.float
        self.model_class = ThetaForecaster
        self.models = {}
        self.n_ts_predictions = n_ts_predictions
        self.ts_analysis = ts_analysis
        self.forecasting_horizon = np.arange(1, self.n_ts_predictions)
        self.cutoff_index = {}  # marks index at which training data stops and forecasting window starts
        self.grouped_by = ['__default'] if not ts_analysis['tss'].group_by else ts_analysis['tss'].group_by

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting sktime forecaster for array prediction')

        all_folds = ConcatedEncodedDs(ds_arr)
        data = {'data': all_folds.data_frame[self.target].reset_index(drop=True),
                'group_info': {gcol: all_folds.data_frame[gcol].tolist() for gcol in self.grouped_by}}

        for group in self.ts_analysis['group_combinations']:
            self.models[group] = self.model_class()
            series_idxs, series_data = get_group_matches(data, group)
            if series_data.size > 0:
                series = pd.Series(series_data.squeeze(), index=series_idxs)
                series = series.sort_index(ascending=True)
                series = series.reset_index(drop=True)
                self.models[group].fit(series)
                self.cutoff_index[group] = len(series)

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

        # print(ydf)

        # ydf['prediction'] = self.model.predict(ds.data_frame[self.target].index).tolist()

        data = {'data': ds.data_frame[self.target].reset_index(drop=True),
                'group_info': {gcol: ds.data_frame[gcol].tolist() for gcol in self.grouped_by}}

        # all_idxs = list(range(length))  # @TODO: substract, and assign empty predictions to remainder

        for group in self.ts_analysis['group_combinations']:
            series_idxs, series_data = get_group_matches(data, group)
            if series_data.size > 0:
                series = pd.Series(series_data.squeeze(), index=series_idxs)
                series = series.sort_index(ascending=True)
                series = series.reset_index(drop=True)
                cutoff = self.cutoff_index[group]
                # ydf['prediction'][series_idxs] = self.models[group].predict(np.arange(cutoff,
                #                                                                       cutoff+self.n_ts_predictions)
                #                                                             ).tolist()

                for idx, _ in enumerate(series.iteritems()):
                    ydf['prediction'].iloc[series_idxs[idx]] = self.models[group].predict(
                        np.arange(idx+cutoff,
                                  idx+cutoff+self.n_ts_predictions)).tolist()
            # ydf[f'prediction_{timestep}'] = self.models[timestep](ds)

        # ydf['prediction'] = ydf.values.tolist()

        # print(ydf[['prediction']])
        # print(ydf[['prediction']].shape)

        return ydf[['prediction']]
