import pandas as pd
from lightwood.api.types import TimeseriesSettings


def get_inferred_timestamps(df: pd.DataFrame, col: str, deltas: dict, tss: TimeseriesSettings) -> pd.DataFrame:
    nr_predictions = tss.nr_predictions
    if tss.group_by:
        gby = [f'group_{g}' for g in tss.group_by]

    for (idx, row) in df.iterrows():
        last = row[f'order_{col}'][-1]

        if tss.group_by:
            try:
                series_delta = deltas[frozenset(row[gby].tolist())][col]
            except KeyError:
                series_delta = deltas['__default'][col]
        else:
            series_delta = deltas['__default'][col]
        timestamps = [last + t * series_delta for t in range(nr_predictions)]

        if tss.nr_predictions == 1:
            timestamps = timestamps[0]  # preserves original input format if nr_predictions == 1

        df[f'order_{col}'].iloc[idx] = timestamps
    return df[f'order_{col}']


def add_tn_conf_bounds(data: pd.DataFrame, tss_args: TimeseriesSettings):
    """
    Add confidence (and bounds if applicable) to t+n predictions, for n>1
    @TODO: active research question: how to guarantee 1-e coverage for t+n, n>1
    for now, we replicate the width and conf obtained for t+1
    """
    for col in ['confidence', 'lower', 'upper']:
        data[col] = data[col].astype(object)

    for idx, row in data.iterrows():
        data['confidence'].iloc[idx] = [row['confidence'] for _ in range(tss_args.nr_predictions)]

        preds = row['prediction']
        width = row['upper'] - row['lower']
        data['lower'].iloc[idx] = [pred - width / 2 for pred in preds]
        data['upper'].iloc[idx] = [pred + width / 2 for pred in preds]

    return data
