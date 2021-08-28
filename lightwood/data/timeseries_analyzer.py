from typing import Dict
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.encoder.time_series.helpers.common import get_group_matches, generate_target_group_normalizers


def timeseries_analyzer(data: pd.DataFrame, dtype_dict: Dict[str, str],
                        timeseries_settings: TimeseriesSettings, target: str) -> (Dict, Dict):
    info = {
        'original_type': dtype_dict[target],
        'data': data[target].values
    }
    if timeseries_settings.group_by is not None:
        info['group_info'] = {gcol: data[gcol].tolist() for gcol in timeseries_settings.group_by}  # group col values
    else:
        info['group_info'] = {}

    new_data = generate_target_group_normalizers(info, timeseries_settings)
    deltas = get_delta(data[timeseries_settings.order_by],
                       info,
                       new_data['group_combinations'],
                       timeseries_settings.order_by)

    return {'target_normalizers': new_data['target_normalizers'],
            'deltas': deltas,
            'tss': timeseries_settings,
            'group_combinations': new_data['group_combinations']}


def get_delta(df: pd.DataFrame, ts_info: dict, group_combinations: list, order_cols: list):
    """
    Infer the sampling interval of each time series
    """
    deltas = {"__default": {}}

    for col in order_cols:
        series = pd.Series([x[-1] for x in df[col]])
        rolling_diff = series.rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])
        delta = rolling_diff.value_counts(ascending=False).keys()[0]
        deltas["__default"][col] = delta

    if ts_info.get('group_info', False):
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

    return deltas
