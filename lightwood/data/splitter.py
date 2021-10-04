# TODO: Make stratification work for regression via histogram bins??
from lightwood.api.dtype import dtype
import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import product
from lightwood.api.types import TimeseriesSettings


def splitter(
    data: pd.DataFrame,
    tss: TimeseriesSettings,
    dtype_dict: Dict[str, str],
    seed: int,
    pct_train: int,
    pct_dev: int,
    pct_test: int,
    target: str
) -> Dict[str, pd.DataFrame]:
    """
    Splits a dataset into stratified training/test. First shuffles the data within the dataframe (via ``df.sample``).

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param pct_train: training fraction of data; must be less than 1
    :param dtype_dict: Dictionary with the data type of all columns
    :param seed: Random state for pandas data-frame shuffling
    :param n_subsets: Number of subsets to create from data (for time-series)
    :param target: Name of the target column; if specified, data will be stratified on this column

    :returns: A dictionary containing the keys train, test and dev with their respective data frames, as well as the "stratified_on" key indicating which columns the data was stratified on (None if it wasn't stratified on anything)
    """ # noqa
    if pct_train + pct_dev + pct_test != 100:
        raise Exception('The train, dev and test percentage of the data needs to sum up to 100')

    # Shuffle the data
    if not tss.is_timeseries:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    stratify_on = None
    if tss.is_timeseries or dtype_dict[target] in (dtype.categorical, dtype.binary) and target is not None:
        stratify_on = [target]
        if isinstance(tss.group_by, list):
            stratify_on = stratify_on + tss.group_by
        subsets = stratify(data, 100, stratify_on)
    else:
        subsets = np.array_split(data, 100)

    train = pd.concat(subsets[0:pct_train])
    dev = pd.concat(subsets[pct_train:pct_train + pct_dev])
    test = pd.concat(subsets[pct_train + pct_dev:])

    return {"train": train, "test": test, "dev": dev, "stratified_on": stratify_on}


def stratify(data: pd.DataFrame, nr_subset: int, stratify_on: List[str]) -> List[pd.DataFrame]:
    """
    Splitter for grouped time series tasks, where there is a set of `gcols` columns by which data is grouped.
    Each group yields a different time series, and the splitter generates `k` subsets from `data`,
    with equally-sized sub-series for each group.

    :param data: Data to be split
    :param nr_subset: Number of subsets to create
    :param stratify_on: Columns to group-by on

    :returns A list of equally-sized data subsets that can be concatenated by the full data. This preserves the group-by columns.
    """  # noqa
    all_group_combinations = list(product(*[data[col].unique() for col in stratify_on]))

    subsets = [pd.DataFrame() for _ in range(nr_subset)]
    for group in all_group_combinations:
        subframe = data
        for idx, col in enumerate(stratify_on):
            subframe = subframe[subframe[col] == group[idx]]

        subset = np.array_split(subframe, nr_subset)

        for i in range(nr_subset):
            subsets[i] = pd.concat([subsets[i], subset[i]])

    return subsets
