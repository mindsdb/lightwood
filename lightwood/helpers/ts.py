import numpy as np
import pandas as pd
from lightwood.api.types import TimeseriesSettings


def get_inferred_timestamps(df: pd.DataFrame, col: str, deltas: dict, tss: TimeseriesSettings) -> pd.DataFrame:
    horizon = tss.horizon
    if tss.group_by:
        gby = [f'group_{g}' for g in tss.group_by]

    for (idx, row) in df.iterrows():
        last = [r for r in row[f'order_{col}'] if r == r][-1]  # filter out nans (safeguard; it shouldn't happen anyway)

        if tss.group_by:
            try:
                series_delta = deltas[tuple(row[gby].tolist())][col]
            except KeyError:
                series_delta = deltas['__default'][col]
        else:
            series_delta = deltas['__default'][col]
        timestamps = [last + t * series_delta for t in range(horizon)]

        if tss.horizon == 1:
            timestamps = timestamps[0]  # preserves original input format if horizon == 1

        df[f'order_{col}'].iloc[idx] = timestamps
    return df[f'order_{col}']


def add_tn_conf_bounds(data: pd.DataFrame, tss_args: TimeseriesSettings):
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
