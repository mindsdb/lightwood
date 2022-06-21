from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
# from sktime.transformations.series.detrend import Deseasonalizer

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.helpers.ts import get_ts_groups, get_delta
from lightwood.encoder.time_series.helpers.common import generate_target_group_normalizers
from lightwood.helpers.ts import Differencer
from lightwood.helpers.ts import get_group_matches


def timeseries_analyzer(data: Dict[str, pd.DataFrame], dtype_dict: Dict[str, str],  # analysis,
                        timeseries_settings: TimeseriesSettings, target: str) -> Dict:
    """
    This module analyzes (pre-processed) time series data and stores a few useful insights used in the rest of Lightwood's pipeline.
    
    :param data: dictionary with the dataset split into train, val, test subsets. 
    :param dtype_dict: dictionary with inferred types for every column.
    :param analysis: output of statistical analysis phase.
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
    deltas, periods, freqs = get_delta(data['train'], dtype_dict, groups, tss)

    normalizers = generate_target_group_normalizers(data['train'], target, dtype_dict, groups, tss)

    if dtype_dict[target] in (dtype.integer, dtype.float, dtype.num_tsarray):
        naive_forecast_residuals, scale_factor = get_grouped_naive_residuals(data['dev'],
                                                                             target,
                                                                             tss,
                                                                             groups)
        differencers = get_differencers(data['train'], target, groups, tss.group_by)
    else:
        naive_forecast_residuals, scale_factor = {}, {}
        differencers = {}

    stl_transforms = get_stls(data['train'], data['dev'], target, periods, groups, tss)

    return {'target_normalizers': normalizers,
            'deltas': deltas,
            'tss': tss,
            'group_combinations': groups,
            'ts_naive_residuals': naive_forecast_residuals,
            'ts_naive_mae': scale_factor,
            'periods': periods,
            'sample_freqs': freqs,
            'stl_transforms': stl_transforms,
            'differencers': differencers
            }


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
        target: str,
        tss: TimeseriesSettings,
        group_combinations: List) -> Tuple[Dict, Dict]:
    """
    Wraps `get_naive_residuals` for a dataframe with multiple co-existing time series.
    """  # noqa
    group_residuals = {}
    group_scale_factors = {}
    for group in group_combinations:
        idxs, subset = get_group_matches(info, group, tss.group_by)
        residuals, scale_factor = get_naive_residuals(subset[target])  # @TODO: pass m once we handle seasonality
        group_residuals[group] = residuals
        group_scale_factors[group] = scale_factor
    return group_residuals, group_scale_factors


def get_differencers(data: pd.DataFrame, target: str, groups: List, group_cols: List):
    differencers = {}
    for group in groups:
        idxs, subset = get_group_matches(data, group, group_cols)
        differencer = Differencer()
        differencer.fit(subset[target].values)
        differencers[group] = differencer
    return differencers


def get_stls(train_df: pd.DataFrame,
             dev_df: pd.DataFrame,
             target: str,
             sps: Dict,
             groups: list,
             tss: TimeseriesSettings
             ) -> Dict[str, object]:
    stls = {}
    for group in groups:
        _, tr_subset = get_group_matches(train_df, group, tss.group_by)
        _, dev_subset = get_group_matches(dev_df, group, tss.group_by)
        tr_subset.index = tr_subset['__mdb_original_index']
        dev_subset.index = dev_subset[f'__mdb_original_{tss.order_by[0]}']
        # dev_subset.index = pd.to_datetime(dev_subset[f'__mdb_original_{tss.order_by[0]}'], unit='s')
        tr_subset = tr_subset[target]
        dev_subset = dev_subset[target]
        # detrender = _pick_detrender(tr_subset, dev_subset)
        # deseasonalizer = _pick_deseasonalizer(tr_subset, dev_subset)

    return stls


def _pick_detrender(tr_subset, dev_subset):
    detrenders = []
    tr_scores = []
    dev_scores = []

    # TODO: pending: move group count and freq inference to before ts_transform,
    #  and impute missing data (as 0.0, doesn't matter)
    #  then enforce this index and freq so that below is fittable and also usable when transforming or inverting
    #  at arbitrary points

    for degree in [1, 2]:
        detrender = Detrender(forecaster=PolynomialTrendForecaster(degree=degree))
        tr_res = detrender.fit_transform(tr_subset.reset_index(drop=True))
        detrenders.append(detrender)
        tr_scores.append(np.sqrt(mean_squared_error(tr_subset, tr_res)))
        dev_res = detrender.transform(dev_subset.reset_index(drop=True))
        dev_scores.append(np.sqrt(mean_squared_error(dev_subset, dev_res)))

    r2s = np.mean([tr_scores, dev_scores], axis=1)
    return detrenders[np.argmax(r2s)]


def _pick_deseasonalizer(tr_subset, dev_subset):
    # deseasonalizers = []
    # tr_r2s = []
    # dev_r2
    return None
