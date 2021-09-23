# TODO: Make stratification work for grouped cols??
# TODO: Make stratification work for regression via histogram bins??

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from itertools import product

from lightwood.api.types import TimeseriesSettings


def splitter(
    data: pd.DataFrame,
    tss: TimeseriesSettings,
    pct_train: float,
    seed: int = 1,
    target: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Splits a dataset into stratified training/test. First shuffles the data within the dataframe (via ``df.sample``).

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param pct_train: training fraction of data; must be less than 1
    :param seed: Random state for pandas data-frame shuffling
    :param target: Name of the target column; if specified, data will be stratified on this column

    :returns: A dictionary containing "train" and "test" splits of the data.
    """
    if pct_train > 1:
        raise Exception(
            f"The value of pct_train ({pct_train}) needs to be between 0 and 1"
        )

    # Time series needs to preserve the sequence
    if tss.is_timeseries:
        train, test = _split_timeseries(data, target, pct_train, tss)

    else:

        # Shuffle the data
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        train, test = stratify(data, target, pct_train)

    return {"train": train_data, "test": test_data, "stratified_on": target}


def stratify(data: pd.DataFrame, pct_train: float, target: Optional[str]):
    """
    Stratify a dataset on a target column; returns a train/test split.
    """
    if target is None:
        Ntrain = int(len(data) * pct_train)
        train, test = data[:Ntrain], data[Ntrain:]
    else:
        train = []
        test = []

        for label, subset in data.groupby(target):

            # Extract, from each label, 
            N = len(subset)
            Ntrain = int(N * pct_train) # Ensure 1 example passed to test

            train.append(subset[:Ntrain])
            test.append(subset[Ntrain:])

        train = pd.concat(train)
        test = pd.concat(test)

    return train, test


def _split_timeseries(
    data: pd.DataFrame,
    pct_train: float,
    tss: TimeseriesSettings,
    target: Optional[str]
):
    """
    Returns a time-series split based on group-by columns or not for time-series.

    Stratification occurs only when grouped-columns are not specified. If they are, this is overridden.

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param pct_train: training fraction of data; must be less than 1
    :param target: Name of data column to stratify on (usually the predicted target)

    :returns Train/test split of the data of interest
    """
    if not tss.group_by:
        train, test = stratify(data, pct_train, target)
    else:
        gcols = tss.group_by
        subsets = grouped_ts_splitter(data, 30, gcols)
    return subsets


def grouped_ts_splitter(
    data: pd.DataFrame,
    k: int,
    gcols: List[str]
) -> List[pd.DataFrame]:
    """
    Splitter for grouped time series tasks, where there is a set of `gcols` columns by which data is grouped.
    Each group yields a different time series, and the splitter generates `k` subsets from `data`,
    with equally-sized sub-series for each group.

    :param data: Data to be split
    :param k: Number of subsets to create
    :param gcols: Columns to group-by on

    :returns A list of equally-sized data subsets that can be concatenated by the full data. This preserves the group-by columns.
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
