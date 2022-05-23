from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.encoder.base import BaseEncoder
from lightwood.mixer.base import BaseMixer
from lightwood.mixer.lightgbm import LightGBM
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class LightGBMArray(BaseMixer):
    """LightGBM-based model, intended for usage in time series tasks."""
    models: List[LightGBM]
    horizon: int
    submodel_stop_after: float
    target: str
    supports_proba: bool

    def __init__(
            self, stop_after: float, target: str, dtype_dict: Dict[str, str],
            input_cols: List[str],
            horizon: int, fit_on_dev: bool, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.submodel_stop_after = stop_after / horizon
        self.target = target
        # dtype_dict[target] = dtype.float  @TODO: figure out if this can be removed
        self.models = [LightGBM(self.submodel_stop_after, target, dtype_dict, input_cols, fit_on_dev,
                                False, target_encoder)
                       for _ in range(horizon)]
        self.horizon = horizon  # for time series tasks, how long is the forecast horizon
        self.supports_proba = False
        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting LGBM models for array prediction')
        original_target_train = deepcopy(train_data.data_frame[self.target])
        original_target_dev = deepcopy(dev_data.data_frame[self.target])

        for timestep in range(self.horizon):
            if timestep > 0:
                train_data.data_frame[self.target] = train_data.data_frame[f'{self.target}_timestep_{timestep}']
                dev_data.data_frame[self.target] = dev_data.data_frame[f'{self.target}_timestep_{timestep}']

            self.models[timestep].fit(train_data, dev_data)  # @TODO: this call could be parallelized

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
        if args.predict_proba:
            log.warning('This model does not output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.horizon)])

        for timestep in range(self.horizon):
            ydf[f'prediction_{timestep}'] = self.models[timestep](ds, args)['prediction']

        ydf['prediction'] = ydf.values.tolist()
        return ydf[['prediction']]
