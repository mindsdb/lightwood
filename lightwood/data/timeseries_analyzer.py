from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.encoder.time_series.helpers.common import generate_target_group_normalizers
from lightwood.helpers.ts import Differencer
from lightwood.helpers.general import get_group_matches


def timeseries_analyzer(data: Dict[str, pd.DataFrame], dtype_dict: Dict[str, str],
                        timeseries_settings: TimeseriesSettings, target: str) -> Dict:
    """
    This module analyzes (pre-processed) time series data and stores a few useful insights used in the rest of Lightwood's pipeline.
    
    :param data: dictionary with the dataset split into train, val, test subsets. 
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
    tss = timeseries_settings

    normalizers, group_combinations = generate_target_group_normalizers(data['train'], target, dtype_dict, tss)

    if dtype_dict[target] in (dtype.integer, dtype.float, dtype.num_tsarray):
        naive_forecast_residuals, scale_factor = get_grouped_naive_residuals(data['dev'], tss, group_combinations)
    else:
        naive_forecast_residuals, scale_factor = {}, {}

    deltas, periods, freqs = get_delta(data['train'], group_combinations, tss)
    differencers = get_differencers(data['train'], group_combinations, tss.group_by)
    # TODO: transform all data splits using differencers, but not here
    # TODO: must revert this analyzer to before the TS transform, because otherwise it's wasted compute to revert
    #  those array-transforms, differentiate, then put them  back together...

    return {'target_normalizers': normalizers,
            'deltas': deltas,
            'tss': tss,
            'group_combinations': group_combinations,
            'ts_naive_residuals': naive_forecast_residuals,
            'ts_naive_mae': scale_factor,
            'periods': periods,
            'sample_freqs': freqs,
            'differencers': differencers
            }


def get_delta(
        df: pd.DataFrame,
        group_combinations: list,
        tss: TimeseriesSettings
) -> Tuple[Dict, Dict, Dict]:
    """
    Infer the sampling interval of each time series, by picking the most popular time interval observed in the training data.
    
    :param df: Dataframe with time series data.
    :param group_combinations: all tuples with distinct values for `TimeseriesSettings.group_by` columns, defining all available time series.
    :param tss: timeseries settings
    
    :return:
    Dictionary with group combination tuples as keys. Values are dictionaries with the inferred delta for each series.
    """  # noqa
    # pd.to_datetime(df, unit='s', origin='unix')
    # pd.infer_freq(pd.DatetimeIndex(pd.to_datetime(df, unit='s', origin='unix')), warn=True)

    # delta compute
    order_col = [f'__mdb_original_{tss.order_by[0]}']
    deltas = {"__default": df[order_col].rolling(window=2).apply(np.diff).value_counts().index[0][0]}
    print(deltas)
    freq, period = detect_freq_period(deltas["__default"], tss)
    periods = {"__default": period}
    freqs = {"__default": freq}

    if tss.group_by:
        for group in group_combinations:
            if group != "__default":
                _, subset = get_group_matches(df, group, tss.group_by)
                if subset.size > 1:
                    deltas[group] = subset[order_col].rolling(window=2).apply(np.diff).value_counts().index[0][0]
                    period, freq = detect_freq_period(deltas[group], tss)
                    periods[group] = period
                    freqs[group] = freq

    return deltas, periods, freqs

    # def _most_popular(sorted_keys: pd.Index):
    #     idx = 0
    #     while sorted_keys[idx] == 0:
    #         if idx == len(sorted_keys) - 1:
    #             break
    #         else:
    #             idx += 1
    #     return sorted_keys[idx]
    #
    # deltas = {"__default": {}}
    #
    # # get default delta for all data
    # for col in order_cols:
    #     series = pd.Series([x[-1] for x in df[col]])
    #     series = series.drop_duplicates()  # by this point df is ordered so duplicate timestamps are either because of non-handled groups or repeated data that, for mode delta estimation, should be ignored  # noqa
    #     rolling_diff = series.rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
    #     delta = _most_popular(rolling_diff.value_counts(ascending=False).keys())  # pick most popular
    #     deltas["__default"][col] = delta

    # # get group-wise deltas (if applicable)
    # if tss.group_by:
    #     # original_data = ts_info['data']
    #     for group in group_combinations:
    #         if group != "__default":
    #             for col in tss.order_by:
    #                 ts_info['data'] = pd.Series([x[-1] for x in df[col]])
    #                 _, subset = get_group_matches(ts_info, group)
    #                 if subset.size > 1:
    #                     rolling_diff = pd.Series(
    #                         subset.squeeze()).rolling(
    #                         window=2).apply(
    #                         lambda x: x.iloc[1] - x.iloc[0])
    #                     delta = _most_popular(rolling_diff.value_counts(ascending=False).keys())
    #                     if group in deltas:
    #                         deltas[group][col] = delta
    #                     else:
    #                         deltas[group] = {col: delta}
    #     ts_info['data'] = original_data

    # return deltas


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
    # @TODO: support categorical series as well
    residuals = target_data.rolling(window=m + 1).apply(lambda x: abs(x.iloc[m] - x.iloc[0]))[m:].values.flatten()
    scale_factor = np.average(residuals)
    return residuals.tolist(), scale_factor


def get_grouped_naive_residuals(
        info: pd.DataFrame,
        tss: TimeseriesSettings,
        group_combinations: List) -> Tuple[Dict, Dict]:
    """
    Wraps `get_naive_residuals` for a dataframe with multiple co-existing time series.
    """  # noqa
    group_residuals = {}
    group_scale_factors = {}
    for group in group_combinations:
        idxs, subset = get_group_matches(info, group, tss.group_by)
        residuals, scale_factor = get_naive_residuals(pd.DataFrame(subset))  # @TODO: pass m once we handle seasonality
        group_residuals[group] = residuals
        group_scale_factors[group] = scale_factor
    return group_residuals, group_scale_factors


def detect_freq_period(deltas: pd.DataFrame, tss: TimeseriesSettings) -> tuple:
    """
    Helper method that, based on the most popular interval for a time series, determines its seasonal peridiocity (sp).
    This bit of information can be crucial for good modelling with methods like ARIMA.

    Supported time intervals are:
        * 'year'
        * 'semestral'
        * 'quarter'
        * 'bimonthly'
        * 'monthly'
        * 'weekly'
        * 'daily'
        * 'hourly'
        * 'minute'
        * 'second'

    Note: all computations assume that the first provided `order_by` column is the one that specifies the sp.

    :param deltas: output of `get_delta`, has the most popular interval for each time series.
    :param tss: timeseries settings.

    :return: for all time series 1) a dictionary with its sp and 2) a dictionary with the detected sampling frequency
    """  # noqa
    secs_to_interval = {
        'year': 60 * 60 * 24 * 365,
        'semestral': 60 * 60 * 24 * 365 // 2,
        'quarter': 60 * 60 * 24 * 365 // 4,
        'bimonthly': 60 * 60 * 24 * 365 // 6,
        'monthly': 60 * 60 * 24 * 31,
        'weekly': 60 * 60 * 24 * 7,
        'daily': 60 * 60 * 24,
        'hourly': 60 * 60,
        'minute': 60,
        'second': 1
    }
    freq_to_period = {interval: period for (interval, period) in tss.interval_periods}
    for tag, period in (('year', 1), ('semestral', 2), ('quarter', 4), ('bimonthly', 6), ('monthly', 12),
                        ('weekly', 4), ('daily', 1), ('hourly', 24), ('minute', 1), ('second', 1)):
        if tag not in freq_to_period.keys():
            freq_to_period[tag] = period

    diffs = [(tag, abs(deltas - secs)) for tag, secs in secs_to_interval.items()]
    freq, min_diff = sorted(diffs, key=lambda x: x[1])[0]
    return freq, freq_to_period.get(freq, 1)


def get_differencers(data: pd.DataFrame, groups: List, group_cols: List):
    differencers = {}
    for group in groups:
        idxs, subset = get_group_matches(data, group, group_cols)
        differencer = Differencer()
        differencer.fit(subset)
        differencers[group] = differencer
    return differencers
