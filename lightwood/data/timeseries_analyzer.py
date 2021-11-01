from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.encoder.time_series.helpers.common import generate_target_group_normalizers
from lightwood.helpers.general import get_group_matches


def timeseries_analyzer(data: pd.DataFrame, dtype_dict: Dict[str, str],
                        timeseries_settings: TimeseriesSettings, target: str) -> Dict:
    """
    This module analyzes (pre-processed) time series data and stores a few useful insights used in the rest of Lightwood's pipeline.
    
    :param data: dataframe with time series dataset. 
    :param dtype_dict: dictionary with inferred types for every column.
    :param timeseries_settings: A `TimeseriesSettings` object. For more details, check `lightwood.types.TimeseriesSettings`.
    :param target: name of the target column.
    
    The following things are extracted from each time series inside the dataset:
      - group_combinations: all observed combinations of values for the set of `group_by` columns. The length of this list determines how many time series are in the data.
      - deltas: inferred sampling interval 
      - ts_naive_residuals: Residuals obtained from the data by a naive forecaster that repeats the last-seen value. 
      - ts_naive_mae: Mean residual value obtained from the data by a naive forecaster that repeats the last-seen value.
      - target_normalizers: objects that may normalize the data within any given time series for effective learning. See `lightwood.encoder.time_series.helpers.common` for available choices.
    
    :return: Dictionary with the aforementioned insights and the `TimeseriesSettings` object for future references.
    """  # noqa
    info = {
        'original_type': dtype_dict[target],
        'data': data[target].values
    }
    if timeseries_settings.group_by is not None:
        info['group_info'] = {gcol: data[gcol].tolist() for gcol in timeseries_settings.group_by}  # group col values
    else:
        info['group_info'] = {}

    # @TODO: maybe normalizers should fit using only the training subsets??
    new_data = generate_target_group_normalizers(info)

    if dtype_dict[target] in (dtype.integer, dtype.float, dtype.tsarray):
        naive_forecast_residuals, scale_factor = get_grouped_naive_residuals(info, new_data['group_combinations'])
    else:
        naive_forecast_residuals, scale_factor = {}, {}

    deltas = get_delta(data[timeseries_settings.order_by],
                       info,
                       new_data['group_combinations'],
                       timeseries_settings.order_by)

    return {'target_normalizers': new_data['target_normalizers'],
            'deltas': deltas,
            'tss': timeseries_settings,
            'group_combinations': new_data['group_combinations'],
            'ts_naive_residuals': naive_forecast_residuals,
            'ts_naive_mae': scale_factor
            }


def get_delta(df: pd.DataFrame, ts_info: dict, group_combinations: list, order_cols: list) -> Dict[str, Dict]:
    """
    Infer the sampling interval of each time series, by picking the most popular time interval observed in the training data.
    
    :param df: Dataframe with time series data.
    :param ts_info: Dictionary used internally by `timeseries_analyzer`. Contains group-wise series information, among other things.
    :param group_combinations: all tuples with distinct values for `TimeseriesSettings.group_by` columns, defining all available time series.
    :param order_cols: all columns specified in `TimeseriesSettings.order_by`. 
    
    :return:
    Dictionary with group combination tuples as keys. Values are dictionaries with the inferred delta for each series, for each `order_col`.
    """  # noqa
    deltas = {"__default": {}}

    # get default delta for all data
    for col in order_cols:
        series = pd.Series([x[-1] for x in df[col]])
        rolling_diff = series.rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
        delta = rolling_diff.value_counts(ascending=False).keys()[0]  # pick most popular
        deltas["__default"][col] = delta

    # get group-wise deltas (if applicable)
    if ts_info.get('group_info', False):
        original_data = ts_info['data']
        for group in group_combinations:
            if group != "__default":
                deltas[group] = {}
                for col in order_cols:
                    ts_info['data'] = pd.Series([x[-1] for x in df[col]])
                    _, subset = get_group_matches(ts_info, group)
                    if subset.size > 1:
                        rolling_diff = pd.Series(
                            subset.squeeze()).rolling(
                            window=2).apply(
                            lambda x: x.iloc[1] - x.iloc[0])
                        delta = rolling_diff.value_counts(ascending=False).keys()[0]
                        deltas[group][col] = delta
        ts_info['data'] = original_data

    return deltas


def get_naive_residuals(target_data: pd.DataFrame, m: int = 1) -> Tuple[List, float]:
    """
    Computes forecasting residuals for the naive method (forecasts for time `t` is the value observed at `t-1`).
    Useful for computing MASE forecasting error.

    Note: method assumes predictions are all for the same group combination. For a dataframe that contains multiple
     series, use `get_grouped_naive_resiudals`.

    :param target_data: observed time series targets
    :param m: season length. the naive forecasts will be the m-th previously seen value for each series

    :return: (list of naive residuals, average residual value)
    """  # noqa
    residuals = target_data.rolling(window=m + 1).apply(lambda x: abs(x.iloc[m] - x.iloc[0]))[m:].values.flatten()
    scale_factor = np.average(residuals)
    return residuals.tolist(), scale_factor


def get_grouped_naive_residuals(info: Dict, group_combinations: List) -> Tuple[Dict, Dict]:
    """
    Wraps `get_naive_residuals` for a dataframe with multiple co-existing time series.
    """  # noqa
    group_residuals = {}
    group_scale_factors = {}
    for group in group_combinations:
        idxs, subset = get_group_matches(info, group)
        residuals, scale_factor = get_naive_residuals(pd.DataFrame(subset))  # @TODO: pass m once we handle seasonality
        group_residuals[group] = residuals
        group_scale_factors[group] = scale_factor
    return group_residuals, group_scale_factors
