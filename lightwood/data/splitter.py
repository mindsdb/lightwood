# TODO: Make stratification work for grouped cols??
# TODO: Make stratification work for regression via histogram bins??

from lightwood import dtype
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from itertools import product
from lightwood.api.types import TimeseriesSettings


def splitter(
    data: pd.DataFrame,
    tss: TimeseriesSettings,
    pct_train: float,
    dtype_dict: Dict[str, str],
    seed: int = 1,
    N_subsets: int = 30,
    target: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Splits a dataset into stratified training/test. First shuffles the data within the dataframe (via ``df.sample``).

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param pct_train: training fraction of data; must be less than 1
    :param dtype_dict: Dictionary with the data type of all columns
    :param seed: Random state for pandas data-frame shuffling
    :param N_subsets: Number of subsets to create from data (for time-series)
    :param target: Name of the target column; if specified, data will be stratified on this column

    :returns: A dictionary containing "train" and "test" splits of the data.
    """
    if pct_train > 1:
        raise Exception(
            f"The value of pct_train ({pct_train}) needs to be between 0 and 1"
        )

    # Time series needs to preserve the sequence
    if tss.is_timeseries:
        train, test = _split_timeseries(data, tss, pct_train, N_subsets)

    else:
        # Shuffle the data
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        if dtype_dict[target] in (dtype.categorical, dtype.binary):
            train, test = stratify(data, target, pct_train)

    return {"train": train, "test": test, "stratified_on": target}


def stratify(data: pd.DataFrame, pct_train: float, target: Optional[str] = None):
    """
    Stratify a dataset on a target column; returns a train/test split.

    :param data: Dataset to split into training/testing
    :param pct_train: Fraction of data reserved for training (rest is testing)
    :param target: Name of the target column to stratify on
    """
    if target is None:
        n_train = int(len(data) * pct_train)
        train, test = data[:n_train], data[n_train:]
    else:
        train = []
        test = []

        for _, subset in data.groupby(target):

            # Extract, from each label,
            n_train = int(len(subset) * pct_train)  # Ensure 1 example passed to test

            train.append(subset[:n_train])
            test.append(subset[n_train:])

        train = pd.concat(train)
        test = pd.concat(test)

    return train, test


def _split_timeseries(
    data: pd.DataFrame,
    tss: TimeseriesSettings,
    pct_train: float,
    k: int = 30,
):
    """
    Returns a time-series split based on group-by columns or not for time-series.

    Stratification occurs only when grouped-columns are not specified. If they are, this is overridden.

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param pct_train: Fraction of data reserved for training
    :param k: Number of subsets to create

    :returns Train/test split of the data
    """
    gcols = tss.group_by
    subsets = grouped_ts_splitter(data, k, gcols)
    return subsets[:int(pct_train * k)], subsets[int(pct_train * k):]


def grouped_ts_splitter(
    data: pd.DataFrame, k: int, gcols: List[str]
) -> List[pd.DataFrame]:
    """
    Splitter for grouped time series tasks, where there is a set of `gcols` columns by which data is grouped.
    Each group yields a different time series, and the splitter generates `k` subsets from `data`,
    with equally-sized sub-series for each group.

    :param data: Data to be split
    :param k: Number of subsets to create
    :param gcols: Columns to group-by on

    :returns A list of equally-sized data subsets that can be concatenated by the full data. This preserves the group-by columns.
    """  # noqa
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
