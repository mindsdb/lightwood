from copy import deepcopy
from typing import Dict, Tuple, List, Union

import optuna
import numpy as np
import pandas as pd
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import ConditionalDeseasonalizer

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.helpers.ts import get_ts_groups, get_delta, get_group_matches, Differencer, max_pacf
from lightwood.helpers.log import log
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
    deltas, periods, freqs = get_delta(data['train'], dtype_dict, groups, tss)

    normalizers = generate_target_group_normalizers(data['train'], target, dtype_dict, groups, tss)

    if dtype_dict[target] in (dtype.integer, dtype.float, dtype.num_tsarray):
        periods = max_pacf(data['train'], groups, target, tss)  # override with PACF output
        naive_forecast_residuals, scale_factor = get_grouped_naive_residuals(data['dev'],
                                                                             target,
                                                                             tss,
                                                                             groups)
        differencers = get_differencers(data['train'], target, groups, tss.group_by)
        stl_transforms = get_stls(data['train'], data['dev'], target, periods, groups, tss)
    else:
        naive_forecast_residuals, scale_factor = {}, {}
        differencers = {}
        stl_transforms = {}

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
        if subset.shape[0] > 1:
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
    stls = {'__default': None}
    for group in groups:
        if group != '__default':
            _, tr_subset = get_group_matches(train_df, group, tss.group_by)
            _, dev_subset = get_group_matches(dev_df, group, tss.group_by)
            if tr_subset.shape[0] > 0 and dev_subset.shape[0] > 0 and sps.get(group, False):
                group_freq = tr_subset['__mdb_inferred_freq'].iloc[0]
                tr_subset = deepcopy(tr_subset)[target]
                dev_subset = deepcopy(dev_subset)[target]
                tr_subset.index = pd.date_range(start=tr_subset.iloc[0], freq=group_freq,
                                                periods=len(tr_subset)).to_period()
                dev_subset.index = pd.date_range(start=dev_subset.iloc[0], freq=group_freq,
                                                 periods=len(dev_subset)).to_period()
                stl = _pick_ST(tr_subset, dev_subset, sps[group])
                log.info(f'Best STL decomposition params for group {group} are: {stl["best_params"]}')
                stls[group] = stl
    return stls


def _pick_ST(tr_subset: pd.Series, dev_subset: pd.Series, sp: list):
    """
    Perform hyperparam search with optuna to find best combination of ST transforms for a time series.

    :param tr_subset: training series used for fitting blocks. Index should be datetime, and values are the actual time series.
    :param dev_subset: dev series used for computing loss. Index should be datetime, and values are the actual time series.
    :param sp: list of candidate seasonal periods
    :return: best deseasonalizer and detrender combination based on dev_loss
    """  # noqa

    def _ST_objective(trial: optuna.Trial):
        trend_degree = trial.suggest_categorical("trend_degree", [1, 2])
        ds_sp = trial.suggest_categorical("ds_sp", sp)  # seasonality period to use in deseasonalizer
        if min(min(tr_subset), min(dev_subset)) <= 0:
            decomp_type = trial.suggest_categorical("decomp_type", ['additive'])
        else:
            decomp_type = trial.suggest_categorical("decomp_type", ['additive', 'multiplicative'])

        detrender = Detrender(forecaster=PolynomialTrendForecaster(degree=trend_degree))
        deseasonalizer = ConditionalDeseasonalizer(sp=ds_sp, model=decomp_type)
        transformer = STLTransformer(detrender=detrender, deseasonalizer=deseasonalizer, type=decomp_type)
        transformer.fit(tr_subset)
        residuals = transformer.transform(dev_subset)

        trial.set_user_attr("transformer", transformer)
        return np.power(residuals, 2).sum()

    space = {"trend_degree": [1, 2], "ds_sp": sp, "decomp_type": ['additive', 'multiplicative']}
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(space))
    study.optimize(_ST_objective, n_trials=8)

    return {
        "transformer": study.best_trial.user_attrs['transformer'],
        "best_params": study.best_params
    }


class STLTransformer:
    def __init__(self, detrender: Detrender, deseasonalizer: ConditionalDeseasonalizer, type: str = 'additive'):
        """
        Class that handles STL transformation and inverse, given specific detrender and deseasonalizer instances.
        :param detrender: Already initialized. 
        :param deseasonalizer: Already initialized. 
        :param type: Either 'additive' or 'multiplicative'.
        """  # noqa
        self._type = type
        self.detrender = detrender
        self.deseasonalizer = deseasonalizer
        self.op = {
            'additive': lambda x, y: x - y,
            'multiplicative': lambda x, y: x / y
        }
        self.iop = {
            'additive': lambda x, y: x + y,
            'multiplicative': lambda x, y: x * y
        }

    def fit(self, x: Union[pd.DataFrame, pd.Series]):
        self.deseasonalizer.fit(x)
        self.detrender.fit(self.op[self._type](x, self.deseasonalizer.transform(x)))

    def transform(self, x: Union[pd.DataFrame, pd.Series]):
        return self.detrender.transform(self.deseasonalizer.transform(x))

    def inverse_transform(self, x: Union[pd.DataFrame, pd.Series]):
        return self.deseasonalizer.inverse_transform(self.detrender.inverse_transform(x))
