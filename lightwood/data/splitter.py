import pandas as pd
import numpy as np
from typing import List
from itertools import product

from lightwood.api.types import TimeseriesSettings


def splitter(data: pd.DataFrame, k: int, tss: TimeseriesSettings, 
             seed: int, pct_train: float) -> List[pd.DataFrame]:
    """
    Splits a dataframe into k equally-sized subsets.
    """
    if pct_train > 1:
        raise Exception(f'The value of pct_train ({pct_train}) needs to be between 0 and 1')

    if not tss.is_timeseries:
        # shuffle
        data = data.sample(frac=1, seed=seed if seed is not None else len(data)).reset_index(drop=True)

        # split
        subsets = np.array_split(data, k)

    else:
        if not tss.group_by:
            subsets = np.array_split(data, k)
        else:
            gcols = tss.group_by
            subsets = grouped_ts_splitter(data, k, gcols)

    train_data = subsets[0:int(k * pct_train)]
    test_data = subsets[int(k * pct_train):]
    return {
        'train': train_data,
        'test': test_data
    }


def grouped_ts_splitter(data: pd.DataFrame, k: int, gcols: List[str]):
    """
    Splitter for grouped time series tasks, where there is a set of `gcols` columns by which data is grouped.
    Each group yields a different time series, and the splitter generates `k` subsets from `data`,
    with equally-sized sub-series for each group.
    """
    all_group_combinations = list(product(*[data[gcol].unique() for gcol in gcols]))
    subsets = [pd.DataFrame() for _ in range(k)]
    for group in all_group_combinations:
        subframe = data
        for idx, gcol in enumerate(gcols):
            subframe = subframe[subframe[gcol] == group[idx]]

        subset = np.array_split(subframe, k)

        for i in range(k):
            subsets[i] = pd.concat([subsets[i], subset[i]])

    return subsets
