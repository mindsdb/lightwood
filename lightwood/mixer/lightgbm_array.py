from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.helpers.ts import get_group_matches, get_ts_groups
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

    def _fit(self, train_data: EncodedDs, dev_data: EncodedDs, submodel_method='fit') -> None:
        original_train = deepcopy(train_data.data_frame)
        original_dev = deepcopy(dev_data.data_frame)

        if self.ts_analysis.get('stl_transforms', False):
            tr_midx = pd.MultiIndex.from_frame(train_data.data_frame.reset_index()[[*self.tss.group_by, 'index']])
            dev_midx = pd.MultiIndex.from_frame(dev_data.data_frame.reset_index()[[*self.tss.group_by, 'index']])
            train_data.data_frame.index = tr_midx
            dev_data.data_frame.index = dev_midx

            for group in self.ts_analysis['group_combinations']:
                if len(self.ts_analysis['group_combinations']) == 1 or group != '__default':
                    train_idxs, train_subset = get_group_matches(train_data.data_frame, group, self.tss.group_by)
                    dev_idxs, dev_subset = get_group_matches(dev_data.data_frame, group, self.tss.group_by)
                    train_data.data_frame[self.target].loc[train_idxs] = self._transform_target(train_subset[self.target], group).values
                    dev_data.data_frame[self.target].loc[dev_idxs] = self._transform_target(dev_subset[self.target], group).values

                    # shift all timestep cols here by respective offset
                    for timestep in range(1, self.horizon):
                        train_data.data_frame[f'{self.target}_timestep_{timestep}'].loc[train_idxs] = train_data.data_frame[self.target].loc[train_idxs].shift(-timestep)
                        dev_data.data_frame[f'{self.target}_timestep_{timestep}'].loc[dev_idxs] = dev_data.data_frame[self.target].loc[dev_idxs].shift(-timestep)

            # afterwards, drop all nans
            # TODO: risk of no valid points...  would have to do this at transform time to solve, not sure if possible!
            train_data.data_frame = train_data.data_frame.dropna()
            dev_data.data_frame = train_data.data_frame.dropna()

        for timestep in range(self.horizon):
            if timestep > 0:
                train_data.data_frame[self.target] = train_data.data_frame[f'{self.target}_timestep_{timestep}']
                dev_data.data_frame[self.target] = dev_data.data_frame[f'{self.target}_timestep_{timestep}']
            getattr(self.models[timestep], submodel_method)(train_data, dev_data)  # call submodel_method to fit

        # restore target
        train_data.data_frame = original_train
        dev_data.data_frame = original_dev

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info('Started fitting LGBM models for array prediction')
        self._fit(train_data, dev_data, submodel_method='fit')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass
        #  log.info('Updating array of LGBM models...')
        #  self._fit(train_data, dev_data, submodel_method='partial_fit')

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            log.warning('This model does not output probability estimates')

        original_df = deepcopy(ds.data_frame)
        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=[f'prediction_{i}' for i in range(self.horizon)])

        if self.ts_analysis.get('stl_transforms', False):
            midx = pd.MultiIndex.from_frame(ds.data_frame.reset_index()[[*self.tss.group_by, 'index']])
            ds.data_frame.index = midx
            ydf.index = midx
            groups = get_ts_groups(ds.data_frame, self.tss)
            for group in groups:
                group = group if group in self.ts_analysis['group_combinations'] else '__default'
                if len(self.ts_analysis['group_combinations']) == 1 or group != '__default':
                    idxs, subset = get_group_matches(ds.data_frame, group, self.tss.group_by)
                    ds.data_frame[self.target].loc[idxs] = self._transform_target(subset[self.target], group).values

        for timestep in range(self.horizon):
            ydf[f'prediction_{timestep}'] = self.models[timestep](ds, args)['prediction'].values

        if self.ts_analysis.get('stl_transforms', False):
            groups = get_ts_groups(ds.data_frame, self.tss)
            for group in groups:
                group = group if group in self.ts_analysis['group_combinations'] else '__default'
                if len(self.ts_analysis['group_combinations']) == 1 or group != '__default':
                    idxs, subset = get_group_matches(ds.data_frame, group, self.tss.group_by)
                    for timestep in range(self.horizon):
                        ydf[f'prediction_{timestep}'].loc[idxs] = self._inverse_transform_target(
                            ydf[f'prediction_{timestep}'].loc[idxs],
                            group
                        ).values
            ydf = ydf.reset_index(drop=True)

        ydf['prediction'] = ydf.values.tolist()
        ds.data_frame = original_df
        return ydf[['prediction']]

    def _transform_target(self, target_df: pd.DataFrame, group: tuple):
        transformer = self.ts_analysis['stl_transforms'][group]['transformer']
        if isinstance(target_df.index, pd.MultiIndex):
            return transformer.transform(target_df.droplevel(0).to_period())
        else:
            return transformer.transform(target_df.to_period())

    def _inverse_transform_target(self, predictions: pd.DataFrame, group: tuple):
        transformer = self.ts_analysis['stl_transforms'][group]['transformer']
        if isinstance(predictions.index, pd.MultiIndex):
            return transformer.inverse_transform(predictions.droplevel(0).to_period())
        else:
            return transformer.inverse_transform(predictions.to_period())
