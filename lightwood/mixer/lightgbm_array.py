from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.helpers.general import get_group_matches
from lightwood.encoder.base import BaseEncoder
from lightwood.mixer.base import BaseMixer
from lightwood.mixer.lightgbm import LightGBM
from lightwood.api.types import PredictionArguments, TimeseriesSettings
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class LightGBMArray(BaseMixer):
    """LightGBM-based model, intended for usage in time series tasks."""
    models: List[LightGBM]
    submodel_stop_after: float
    target: str
    supports_proba: bool
    ts_analysis: Dict
    tss: TimeseriesSettings

    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            input_cols: List[str],
            fit_on_dev: bool,
            target_encoder: BaseEncoder,
            ts_analysis: Dict[str, object],
            tss: TimeseriesSettings
    ):
        super().__init__(stop_after)
        self.tss = tss
        self.horizon = tss.horizon
        self.submodel_stop_after = stop_after / self.horizon
        self.target = target
        self.models = [LightGBM(self.submodel_stop_after,
                                target,
                                dtype_dict,
                                input_cols,
                                fit_on_dev,
                                False,  # use_optuna
                                target_encoder)
                       for _ in range(self.horizon)]
        self.ts_analysis = ts_analysis
        self.supports_proba = False
        self.stable = True

        if self.ts_analysis['differencers'].get('__default', False):
            for model in self.models:
                model.positive_domain = False  # when differencing, forecasts can be negative even if the domain is not

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting LGBM models for array prediction')
        original_target_train = deepcopy(train_data.data_frame[self.target])
        original_target_dev = deepcopy(dev_data.data_frame[self.target])

        for timestep in range(self.horizon):
            if timestep > 0:
                train_data.data_frame[self.target] = train_data.data_frame[f'{self.target}_timestep_{timestep}']
                dev_data.data_frame[self.target] = dev_data.data_frame[f'{self.target}_timestep_{timestep}']

            # differentiate
            for split in [train_data, dev_data]:
                for group in self.ts_analysis['group_combinations']:
                    idxs, subset = get_group_matches(split.data_frame, group, self.ts_analysis['tss'].group_by)
                    differencer = self.ts_analysis['differencers'].get(group, False)
                    if differencer:
                        split.data_frame.at[idxs, self.target] = differencer.transform(subset[self.target])

            self.models[timestep].fit(train_data, dev_data)

        # restore target
        train_data.data_frame[self.target] = original_target_train
        dev_data.data_frame[self.target] = original_target_dev

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Updating array of LGBM models...')
        original_target_train = deepcopy(train_data.data_frame[self.target])
        original_target_dev = deepcopy(dev_data.data_frame[self.target])

        for timestep in range(self.horizon):
            if timestep > 0:
                train_data.data_frame[self.target] = train_data.data_frame[f'{self.target}_timestep_{timestep}']
                dev_data.data_frame[self.target] = dev_data.data_frame[f'{self.target}_timestep_{timestep}']

            self.models[timestep].partial_fit(train_data, dev_data)  # @TODO: this call could be parallelized

        # restore target
        train_data.data_frame[self.target] = original_target_train
        dev_data.data_frame[self.target] = original_target_dev

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        # TODO: difference the history input (set the last seen value so that we can invert transform predictions
        if args.predict_proba:
            log.warning('This model does not output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.horizon)])

        # TODO: test, enforce!
        for i in range(len(self.models)):
            self.models[i].positive_domain = False

        for timestep in range(self.horizon):
            ydf[f'prediction_{timestep}'] = self.models[timestep](ds, args)['prediction']

        # consolidate if differenced
        for group in self.ts_analysis['group_combinations']:
            differencer = self.ts_analysis['differencers'].get(group, False)
            if differencer:
                idxs, subset = get_group_matches(ds.data_frame.reset_index(drop=True), group, self.ts_analysis['tss'].group_by)
                if subset.size > 1:
                    last_values = [t[-1] for t in subset[f'__mdb_ts_previous_{self.target}']]
                    last_values = [0 if t is None else t for t in last_values]
                    ydf.at[idxs, 'prediction_0'] = ydf.iloc[idxs][f'prediction_0'] + last_values  # TODO this should be call to inverse_transform instead
                    for timestep in range(1, self.horizon):
                        ydf.at[idxs, f'prediction_{timestep}'] = ydf.iloc[idxs][f'prediction_{timestep}'] + ydf.iloc[idxs][f'prediction_{timestep-1}']

        ydf['prediction'] = ydf.values.tolist()
        return ydf[['prediction']]
