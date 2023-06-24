"""
Time series utility methods for usage within mixers.
"""
from typing import Dict
from copy import deepcopy

import pandas as pd


def _transform_target(ts_analysis: Dict[str, Dict], df: pd.DataFrame, freq, group: tuple):
    df = deepcopy(df)  # needed because methods like get_level_values actually have undisclosed side effects  # noqa
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(df.index, pd.MultiIndex) and len(df.index.levels) > 1:
        temp_s = df.droplevel(0)
        temp_s.index = pd.date_range(start=temp_s.index[0], freq=freq, periods=len(temp_s))
        return transformer.transform(temp_s.to_period())
    elif isinstance(df.index, pd.MultiIndex):
        temp_s = pd.Series(data=df.values)
        temp_s.index = df.index.levels[0]  # warning: do not use get_level_values as it removes inferred freq
        return transformer.transform(temp_s.to_period())
    else:
        return transformer.transform(df.to_period())


def _inverse_transform_target(ts_analysis: Dict[str, Dict], predictions: pd.DataFrame, freq, group: tuple):
    predictions = deepcopy(predictions)  # needed because methods like get_level_values actually have undisclosed side effects  # noqa
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(predictions.index, pd.MultiIndex) and len(predictions.index.levels) > 1:
        temp_s = predictions.droplevel(0)
        temp_s.index = pd.date_range(start=temp_s.index[0], freq=freq, periods=len(temp_s))
        return transformer.inverse_transform(temp_s.to_period())
    elif isinstance(predictions.index, pd.MultiIndex):
        temp_s = pd.Series(data=predictions)
        temp_s.index = predictions.index.levels[0]  # warning: do not use get_level_values as it removes inferred freq
        return transformer.inverse_transform(temp_s.to_period())
    else:
        return transformer.inverse_transform(predictions.to_period())
