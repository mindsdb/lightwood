import pandas as pd
from lightwood.api.types import TimeseriesSettings


def get_inferred_timestamps(df: pd.DataFrame, col: str, deltas: dict, tss: TimeseriesSettings) -> pd.DataFrame:
    nr_predictions = tss.nr_predictions
    gby = [f'group_{g}' for g in tss.group_by]

    for (idx, row) in df.iterrows():
        last = row[f'order_{col}'][-1]
        series_delta = deltas[frozenset(row[gby].tolist())][col]
        timestamps = [last + t*series_delta for t in range(nr_predictions)]
        df[f'order_{col}'].iloc[idx] = timestamps
    return df[f'order_{col}']


def add_tn_conf_bounds(data: pd.DataFrame, tss_args: TimeseriesSettings, target_cols, dtypes):
    """
    Add confidence (and bounds if applicable) to t+n predictions, for n>1
        @TODO: active research question: how to guarantee 1-e coverage for t+n, n>1
        for now, we replicate the width and conf obtained for t+1
    """
    for idx in range(len(data[tss_args.order_by[0]])):
        for target in target_cols:
            conf = data[f'{target}_confidence'][idx]
            data[f'{target}_confidence'][idx] = [conf for _ in range(tss_args.nr_predictions)]

            if dtypes[target]['numerical']:
                width = data[f'{target}_confidence_range'][idx][1] - data[f'{target}_confidence_range'][idx][0]
                data[f'{target}_confidence_range'][idx] = [[pred - width / 2, pred + width / 2] for pred in
                                                           data[target][idx]]
    return data