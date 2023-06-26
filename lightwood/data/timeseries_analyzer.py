from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from type_infer.dtype import dtype

from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_ts_groups, get_delta, Differencer
from lightwood.encoder.time_series.helpers.common import generate_target_group_normalizers


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
    groups = get_ts_groups(data['train'], tss)
    deltas, periods, freqs = get_delta(data['train'], tss)

    normalizers = generate_target_group_normalizers(data['train'], target, dtype_dict, tss)

    if dtype_dict[target] in (dtype.integer, dtype.float, dtype.num_tsarray):
        naive_forecast_residuals, scale_factor = get_grouped_naive_residuals(data['dev'], target, tss)
        differencers = get_differencers(data['train'], target, tss.group_by)
    else:
        naive_forecast_residuals, scale_factor = {}, {}
        differencers = {}

    return {'target_normalizers': normalizers,
            'deltas': deltas,
            'tss': tss,
            'group_combinations': groups,
            'ts_naive_residuals': naive_forecast_residuals,
            'ts_naive_mae': scale_factor,
            'periods': periods,
            'sample_freqs': freqs,
            'stl_transforms': {},  # TODO: remove, or provide from outside as user perhaps
            'differencers': differencers
            }


def get_naive_residuals(target_data: pd.DataFrame, m: int = 1) -> Tuple[List, float]:
    """
    Computes forecasting residuals for the naive method (forecasts for time `t` is the value observed at `t-1`).
    Useful for computing MASE forecasting error.

    As per arxiv.org/abs/2203.10716, we resort to a constant forecast based on the last-seen measurement across the entire horizon.
    By following the original measure, the naive forecaster would have the advantage of knowing the actual values whereas the predictor would not.

    Note: method assumes predictions are all for the same group combination. For a dataframe that contains multiple
     series, use `get_grouped_naive_resiudals`.

    :param target_data: observed time series targets
    :param m: season length. the naive forecasts will be the m-th previously seen value for each series

    :return: (list of naive residuals, average residual value)
    """  # noqa
    # @TODO: support categorical series as well
    residuals = np.abs(target_data.values[1:] - target_data.values[0]).flatten()
    scale_factor = np.average(residuals)
    return residuals.tolist(), scale_factor


def get_grouped_naive_residuals(
        info: pd.DataFrame,
        target: str,
        tss: TimeseriesSettings
) -> Tuple[Dict, Dict]:
    """
    Wraps `get_naive_residuals` for a dataframe with multiple co-existing time series.
    """  # noqa
    group_residuals = {}
    group_scale_factors = {}
    grouped = info.groupby(by=tss.group_by) if tss.group_by else info.groupby(lambda x: '__default')
    for group, subset in grouped:
        if subset.shape[0] > 1:
            residuals, scale_factor = get_naive_residuals(subset[target])  # @TODO: pass m once we handle seasonality
            group_residuals[group] = residuals
            group_scale_factors[group] = scale_factor
    return group_residuals, group_scale_factors


def get_differencers(data: pd.DataFrame, target: str, group_cols: List):
    differencers = {}
    grouped = data.groupby(by=group_cols) if group_cols else data.groupby(lambda x: True)
    for group, subset in grouped:
        differencer = Differencer()
        differencer.fit(subset[target].values)
        differencers[group] = differencer
    return differencers
