from typing import Dict
import pandas as pd

from lightwood.api import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.encoder.time_series.helpers.common import MinMaxNormalizer, CatNormalizer, get_group_matches, generate_target_group_normalizers


def timeseries_analyzer(data: pd.DataFrame, dtype_dict: Dict[str, str], timeseries_settings: TimeseriesSettings, target: str) -> (Dict, Dict):
    tss = timeseries_settings
    target_normalizers = {}
    group_combinations = {'__default': data.index.tolist()}

    # instance and train target normalizers
    if dtype_dict[target] in (dtype.categorical, dtype.binary):
        target_normalizers['__default'] = CatNormalizer()
    else:
        target_normalizers['__default'] = MinMaxNormalizer()

    target_normalizers['__default'].prepare(data[target].values)

    # get groups (if any) from training data
    if tss.group_by:
        get_group_matches(data[target].values, tss.group_by, )

    return {'target_normalizers': target_normalizers,
            'group_combinations': group_combinations}