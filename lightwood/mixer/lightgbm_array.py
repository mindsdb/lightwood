from copy import deepcopy
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.mixer.helpers.ts import _apply_stl_on_training, _stl_transform, _stl_inverse_transform
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
            use_stl: bool,
            tss: TimeseriesSettings
    ):
        super().__init__(stop_after)
        self.tss = tss
        self.horizon = tss.horizon
        self.submodel_stop_after = stop_after / self.horizon
        self.target = target
        self.offset_pred_cols = [f'{self.target}_timestep_{i}' for i in range(1, self.horizon)]
        if set(input_cols) != {self.tss.order_by}:
            input_cols.remove(self.tss.order_by)
        for col in self.offset_pred_cols:
            dtype_dict[col] = dtype_dict[self.target]
        self.models = [LightGBM(self.submodel_stop_after,
                                target_col,
                                dtype_dict,
                                input_cols,
                                False,  # fit_on_dev,
                                True if tss.horizon < 10 else False,  # use_optuna
                                target_encoder)
                       for _, target_col in zip(range(self.horizon), [target] + self.offset_pred_cols)]
        self.ts_analysis = ts_analysis
        self.supports_proba = False
        self.use_stl = False
        self.stable = True

    def _fit(self, train_data: EncodedDs, dev_data: EncodedDs, submodel_method='fit') -> None:
        original_train = deepcopy(train_data.data_frame)
        original_dev = deepcopy(dev_data.data_frame)

        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            _apply_stl_on_training(train_data, dev_data, self.target, self.tss, self.ts_analysis)

        for timestep in range(self.horizon):
            getattr(self.models[timestep], submodel_method)(train_data, dev_data)

        # restore dfs
        train_data.data_frame = original_train
        dev_data.data_frame = original_dev

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting LGBM models for array prediction')
        self._fit(train_data, dev_data, submodel_method='fit')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs, args: Optional[dict] = None) -> None:
        log.info('Updating array of LGBM models...')
        self._fit(train_data, dev_data, submodel_method='partial_fit')

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            log.warning('This model does not output probability estimates')

        original_df = deepcopy(ds.data_frame)
        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.horizon)])

        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            ds.data_frame = _stl_transform(ydf, ds, self.target, self.tss, self.ts_analysis)

        for timestep in range(self.horizon):
            ydf[f'prediction_{timestep}'] = self.models[timestep](ds, args)['prediction'].values

        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            ydf = _stl_inverse_transform(ydf, ds, self.tss, self.ts_analysis)

        if self.models[0].positive_domain:
            ydf = ydf.clip(0)

        ydf['prediction'] = ydf.values.tolist()
        ds.data_frame = original_df
        return ydf[['prediction']]
