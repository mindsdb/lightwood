from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf


def get_ts_groups(df: pd.DataFrame, tss) -> list:
    group_combinations = ['__default']
    if tss.group_by:
        groups = [tuple([g]) if not isinstance(g, tuple) else g
                  for g in list(df.groupby(by=tss.group_by).groups.keys())]
        group_combinations.extend(groups)
    return group_combinations


def get_group_matches(
        data: Union[pd.Series, pd.DataFrame],
        combination: tuple,
        group_columns: List[str]
) -> Tuple[list, pd.DataFrame]:
    """Given a particular group combination, return the data subset that belongs to it."""

    if type(data) == pd.Series:
        data = pd.DataFrame(data)
    elif type(data) != pd.DataFrame:
        raise Exception(f"Wrong data type {type(data)}, must be pandas.DataFrame or pd.Series")

    if combination == '__default':
        return list(data.index), data
    else:
        subset = data
        for val, col in zip(combination, group_columns):
            subset = subset[subset[col] == val]
        if len(subset) > 0:
            return list(subset.index), subset
        else:
            return [], pd.DataFrame()


def get_delta(
        df: pd.DataFrame,
        dtype_dict: dict,
        group_combinations: list,
        tss
) -> Tuple[Dict, Dict, Dict]:
    """
    Infer the sampling interval of each time series, by picking the most popular time interval observed in the training data.

    :param df: Dataframe with time series data.
    :param group_combinations: all tuples with distinct values for `TimeseriesSettings.group_by` columns, defining all available time series.
    :param tss: timeseries settings

    :return:
    Dictionary with group combination tuples as keys. Values are dictionaries with the inferred delta for each series.
    """  # noqa
    df = df.copy()
    original_col = f'__mdb_original_{tss.order_by}'
    order_col = original_col if original_col in df.columns else tss.order_by
    deltas = {"__default": df[order_col].astype(float).rolling(window=2).apply(np.diff).value_counts().index[0]}
    freq, period = detect_freq_period(deltas["__default"], tss)
    periods = {"__default": period}
    freqs = {"__default": freq}

    if tss.group_by:
        for group in group_combinations:
            if group != "__default":
                _, subset = get_group_matches(df, group, tss.group_by)
                if subset.shape[0] > 1:
                    deltas[group] = subset[order_col].rolling(window=2).apply(np.diff).value_counts().index[0]
                    freq, period = detect_freq_period(deltas[group], tss)
                    periods[group] = period
                    freqs[group] = freq
                else:
                    deltas[group] = 1.0
                    periods[group] = 1
                    freqs[group] = 'S'

    return deltas, periods, freqs


def get_inferred_timestamps(df: pd.DataFrame, col: str, deltas: dict, tss) -> pd.DataFrame:
    horizon = tss.horizon
    if tss.group_by:
        gby = [f'group_{g}' for g in tss.group_by]

    for (idx, row) in df.iterrows():
        last = [r for r in row[f'order_{col}'] if r == r][-1]  # filter out nans (safeguard; it shouldn't happen anyway)

        if tss.group_by:
            try:
                series_delta = deltas[tuple(row[gby].tolist())]
            except KeyError:
                series_delta = deltas['__default']
        else:
            series_delta = deltas['__default']
        timestamps = [last + t * series_delta for t in range(horizon)]

        if tss.horizon == 1:
            timestamps = timestamps[0]  # preserves original input format if horizon == 1

        df[f'order_{col}'].iloc[idx] = timestamps
    return df[f'order_{col}']


def add_tn_num_conf_bounds(data: pd.DataFrame, tss_args):
    """
    Add confidence (and bounds if applicable) to t+n predictions, for n>1
    TODO: active research question: how to guarantee 1-e coverage for t+n, n>1
    For now, (conservatively) increases width by the confidence times the log of the time step (and a scaling factor).
    """
    for col in ['confidence', 'lower', 'upper']:
        data[col] = data[col].astype(object)

    for idx, row in data.iterrows():
        error_increase = [row['confidence']] + \
                         [row['confidence'] * np.log(np.e + t / 2)  # offset by e so that y intercept is 1
                          for t in range(1, tss_args.horizon)]
        data['confidence'].iloc[idx] = [row['confidence'] for _ in range(tss_args.horizon)]

        preds = row['prediction']
        width = row['upper'] - row['lower']
        data['lower'].iloc[idx] = [pred - (width / 2) * modifier for pred, modifier in zip(preds, error_increase)]
        data['upper'].iloc[idx] = [pred + (width / 2) * modifier for pred, modifier in zip(preds, error_increase)]

    return data


def add_tn_cat_conf_bounds(data: pd.DataFrame, tss_args):
    data['confidence'] = data['confidence'].astype(object)
    for idx, row in data.iterrows():
        data['confidence'].iloc[idx] = [row['confidence'] for _ in range(tss_args.horizon)]
    return data


class Differencer:
    def __init__(self):
        self.original_train_series = None
        self.diffed_train_series = None
        self.first_train_value = None
        self.last_train_value = None

    def diff(self, series: np.array) -> pd.Series:
        series = self._flatten_series(series)
        s = pd.Series(series)
        return s.shift(-1) - s

    def fit(self, series: np.array) -> None:
        series = self._flatten_series(series)
        self.first_train_value = series[0]
        self.last_train_value = series[-1]
        self.original_train_series = series
        self.diffed_train_series = self.diff(series)

    def transform(self, series: np.array) -> pd.Series:
        series = self._flatten_series(series)
        return self.diff(series).shift(1).fillna(0)

    def inverse_transform(self, series: pd.Series, init=None) -> pd.Series:
        origin = init if init else self.last_train_value
        s = pd.Series(origin)
        s = s.append(series).dropna()
        return s.expanding().sum()

    @staticmethod
    def _flatten_series(series: np.ndarray) -> np.ndarray:
        if len(series.shape) > 2:
            raise Exception(f"Input data should be shaped (A,) or (A, 1), got {series.shape}")
        elif len(series.shape) == 2:
            series = series.flatten()
        return series


def detect_freq_period(deltas: pd.DataFrame, tss) -> tuple:
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
        'yearly': 60 * 60 * 24 * 365,
        'quarterly': 60 * 60 * 24 * 365 // 4,
        'bimonthly': 60 * 60 * 24 * 31 * 2,
        'monthly': 60 * 60 * 24 * 31,
        'weekly': 60 * 60 * 24 * 7,
        'daily': 60 * 60 * 24,
        'hourly': 60 * 60,
        'minute': 60,
        'second': 1,
        'millisecond': 0.001,
        'microsecond': 1e-6,
        'nanosecond': 1e-9,
        'constant': 0
    }
    freq_to_period = {interval: period for (interval, period) in tss.interval_periods}
    for tag, period in (('yearly', 1), ('quarterly', 4), ('bimonthly', 6), ('monthly', 12),
                        ('weekly', 4), ('daily', 1), ('hourly', 24), ('minute', 1), ('second', 1), ('constant', 0)):
        if tag not in freq_to_period.keys():
            freq_to_period[tag] = period

    diffs = [(tag, abs(deltas - secs)) for tag, secs in secs_to_interval.items()]
    freq, min_diff = sorted(diffs, key=lambda x: x[1])[0]
    return freq_to_pandas(freq), freq_to_period.get(freq, 1)


def freq_to_pandas(freq, sample_row=None):
    mapping = {
        'constant': 'N',
        'nanosecond': 'N',
        'microsecond': 'us',
        'millisecond': 'ms',
        'second': 'S',
        'minute': 'T',
        'hourly': 'H',  # custom logic
        'daily': 'D',  # custom logic
        'weekly': 'W',  # anchor logic
        'monthly': 'M',  # custom logic
        'bimonthly': 'M',
        'quarterly': 'Q',  # anchor and custom logic
        'yearly': 'Y',  # anchor and custom logic
    }

    # TODO: implement custom dispatch for better precision, use row sample if available:
    #  pandas.pydata.org/docs/user_guide/timeseries.html
    return mapping[freq]


def max_pacf(data: pd.DataFrame, group_combinations, target, tss):
    def min_k(top_k, data):
        return min(top_k, len(data))

    top_k = 5
    k = min_k(top_k, data[target])
    candidate_sps = {'__default': (1 + np.argpartition(pacf(data[target].values)[1:], -k))[-k:].tolist()[::-1]}
    if tss.group_by:
        for group in group_combinations:
            if group != "__default":
                _, subset = get_group_matches(data, group, tss.group_by)
                try:
                    k = min_k(top_k, subset[target])
                    candidates = (1 + np.argpartition(pacf(subset[target].values)[1:], -k))[-k:].tolist()[::-1]
                    candidate_sps[group] = candidates
                except Exception:
                    candidate_sps[group] = None
            if not candidate_sps[group]:
                candidate_sps[group] = [1]

    return candidate_sps
