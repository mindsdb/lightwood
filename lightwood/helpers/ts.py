from typing import Tuple, Dict
from datetime import datetime

import numpy as np
import pandas as pd


def get_ts_groups(df: pd.DataFrame, tss) -> list:
    group_combinations = ['__default']
    if tss.group_by:
        groups = [tuple([g]) if not isinstance(g, tuple) else g
                  for g in list(df.groupby(by=tss.group_by).groups.keys())]
        group_combinations.extend(groups)
    return group_combinations


def get_delta(df: pd.DataFrame, tss) -> Tuple[Dict, Dict, Dict]:
    """
    Infer the sampling interval of each time series, by picking the most popular time interval observed in the training data.

    :param df: Dataframe with time series data.
    :param tss: timeseries settings

    :return:
    Dictionary with group combination tuples as keys. Values are dictionaries with the inferred delta for each series.
    """  # noqa
    df = df.copy()  # TODO: necessary?
    original_col = f'__mdb_original_{tss.order_by}'
    order_col = original_col if original_col in df.columns else tss.order_by
    deltas = {"__default": df[order_col].astype(float).diff().value_counts().index[0]}
    freq, period = detect_freq_period(deltas["__default"], tss, len(df))
    periods = {"__default": [period]}
    freqs = {"__default": freq}

    if tss.group_by:
        grouped = df.groupby(by=tss.group_by)
        for group, subset in grouped:
            if subset.shape[0] > 1:
                deltas[group] = subset[order_col].diff().value_counts().index[0]
                freq, period = detect_freq_period(deltas[group], tss, len(subset))
                freqs[group] = freq
                periods[group] = [period] if period is not None else [1]
            else:
                deltas[group] = 1.0
                periods[group] = [1]
                freqs[group] = 'S'

    return deltas, periods, freqs


def get_inferred_timestamps(df: pd.DataFrame, col: str, deltas: dict, tss, stat_analysis,
                            time_format='') -> pd.DataFrame:
    horizon = tss.horizon

    last = np.vstack(df[f'order_{col}'].dropna().values)[:, -1]

    if tss.group_by:
        gby = [f'group_{g}' for g in tss.group_by]
        series_delta = df[gby].apply(lambda x: deltas.get(tuple(x.values.tolist()),
                                                          deltas['__default']), axis=1).values
        series_delta = series_delta.reshape(-1, 1)
    else:
        series_delta = np.full_like(df.values[:, 0:1], deltas['__default'])

    last = np.repeat(np.expand_dims(last, axis=1), horizon, axis=1)
    lins = np.linspace(0, horizon - 1, num=horizon)
    series_delta = np.repeat(series_delta, horizon, axis=1)
    timestamps = last + series_delta * lins

    if time_format:
        if time_format.lower() == 'infer':
            tformat = stat_analysis.ts_stats['order_format']
        else:
            tformat = time_format

        if tformat:
            def _strfts(elt):
                return datetime.utcfromtimestamp(elt).strftime(tformat)
            timestamps = np.vectorize(_strfts)(timestamps)

    # truncate to horizon
    timestamps = timestamps[:, :horizon]

    # preserves original input format if horizon == 1
    if tss.horizon == 1:
        timestamps = timestamps.squeeze()

    df[f'order_{col}'] = timestamps.tolist()
    return df[f'order_{col}']


def add_tn_num_conf_bounds(data: pd.DataFrame, tss_args):
    """
    Deprecated. Instead we now opt for the much better solution of having scores for each timestep (see all TS classes in analysis/nc)
    
    Add confidence (and bounds if applicable) to t+n predictions, for n>1
    TODO: active research question: how to guarantee 1-e coverage for t+n, n>1
    For now, (conservatively) increases width by the confidence times the log of the time step (and a scaling factor).
    """  # noqa
    for col in ['confidence', 'lower', 'upper']:
        data[col] = data[col].astype(object)

    for idx, row in data.iterrows():
        error_increase = [row['confidence'][0]] + \
                         [row['confidence'][0] * np.log(np.e + t / 2)  # offset by e so that y intercept is 1
                          for t in range(1, tss_args.horizon)]
        data['confidence'].iloc[idx] = [row['confidence'][0] for _ in range(tss_args.horizon)]

        preds = row['prediction']
        width = row['upper'][0] - row['lower'][0]
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


def detect_freq_period(deltas: pd.DataFrame, tss, n_points) -> tuple:
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
                        ('weekly', 52), ('daily', 7), ('hourly', 24), ('minute', 1), ('second', 1), ('constant', 0)):
        if tag not in freq_to_period.keys():
            if period <= n_points:
                freq_to_period[tag] = period
            else:
                freq_to_period[tag] = None

    diffs = [(tag, abs(deltas - secs)) for tag, secs in secs_to_interval.items()]
    freq, min_diff = sorted(diffs, key=lambda x: x[1])[0]
    multiplier = 1
    if secs_to_interval[freq]:
        multiplier += int(min_diff / secs_to_interval[freq])
    return freq_to_pandas(freq, multiplier=multiplier), freq_to_period.get(freq, 1)


def freq_to_pandas(freq, multiplier=1):
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
    items = [mapping[freq]]
    if multiplier > 1:
        items.insert(0, str(multiplier))
    return ''.join(items)


def filter_ts(df: pd.DataFrame, tss, n_rows=1):
    """
    This method triggers only for timeseries datasets.

    It returns a dataframe that filters out all but the first ``n_rows`` per group.
    """  # noqa
    if tss.is_timeseries:
        gby = tss.group_by
        if gby is None:
            df = df.iloc[[0]]
        else:
            ndf = pd.DataFrame(columns=df.columns)
            grouped = df.groupby(by=tss.group_by)
            for group, subdf in grouped:
                if group != '__default':
                    ndf = pd.concat([ndf, subdf.iloc[:n_rows]])
            df = ndf
    return df
