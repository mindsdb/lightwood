from itertools import product
from typing import Dict, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import TransformedTargetForecaster

from lightwood.helpers.log import log
from lightwood.mixer.sktime import SkTime
from lightwood.api.types import PredictionArguments
from lightwood.helpers.general import get_group_matches
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class ProphetMixer(SkTime):
    def __init__(self,
                 stop_after: float,
                 target: str,
                 dtype_dict: Dict[str, str],
                 n_ts_predictions: int,
                 ts_analysis: Dict,
                 model_path: str = 'fbprophet.Prophet',
                 auto_size: bool = True,
                 hyperparam_search: bool = False,
                 target_transforms: Dict[str, Union[int, str]] = {}
                 ):
        super().__init__(stop_after, target, dtype_dict, n_ts_predictions, ts_analysis,
                         model_path, auto_size, hyperparam_search, target_transforms)
        self.stable = False
        self.model_path = model_path
        self.possible_models = [self.model_path]
        self.n_trials = len(self.possible_models)

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            log.warning('This mixer does not output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=['prediction'],
                           dtype=object)

        data = {'data': ds.data_frame,
                'group_info': {gcol: ds.data_frame[gcol].tolist()
                               for gcol in self.grouped_by} if self.ts_analysis['tss'].group_by else {}}

        # @TODO: is this a problem downstream? it does rewrite inside DS too. instead, rewrite .index = np.arange()
        data['data'].reset_index(drop=True, inplace=True)

        pending_idxs = set(range(length))
        all_group_combinations = list(product(*[set(x) for x in data['group_info'].values()]))
        for group in all_group_combinations:
            series_idxs, series_data = get_group_matches(data, group)

            if series_data.size > 0:
                series_data = series_data[:, 0]

                group = frozenset(group)
                series_idxs = sorted(series_idxs)
                if self.models.get(group, False) and self.models[group].is_fitted:
                    forecaster = self.models[group]
                else:
                    log.warning(
                        f"Applying default forecaster for novel group {group}. Performance might not be optimal.")  # noqa
                    forecaster = self.models['__default']
                series = pd.Series(series_data.squeeze(), index=series_idxs)

                ydf = self._call_groupmodel(ydf, forecaster, series, offset=args.forecast_offset)
                pending_idxs -= set(series_idxs)

        # apply default model in all remaining novel-group rows
        if len(pending_idxs) > 0:
            series = data['data'].values
            series = series[:, 0]
            series = pd.Series(series[list(pending_idxs)].squeeze(), index=sorted(list(pending_idxs)))
            ydf = self._call_groupmodel(ydf, self.models['__default'], series, offset=args.forecast_offset)

        return ydf[['prediction']]
