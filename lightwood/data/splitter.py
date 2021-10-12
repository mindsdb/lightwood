from typing import List, Dict, Tuple
from itertools import product

import numpy as np
import pandas as pd

from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings


def splitter(
    data: pd.DataFrame,
    tss: TimeseriesSettings,
    dtype_dict: Dict[str, str],
    seed: int,
    pct_train: float,
    pct_dev: float,
    pct_test: float,
    target: str
) -> Dict[str, pd.DataFrame]:
    """
    Splits data into training, dev and testing datasets. 
    
    Rows in the dataset are shuffled randomly. If a target value is provided and is of data type categorical/binary, then train/test/dev will be stratified to maintain the representative populations of each class.

    :param data: Input dataset to be split
    :param tss: time-series specific details for splitting
    :param dtype_dict: Dictionary with the data type of all columns
    :param seed: Random state for pandas data-frame shuffling
    :param pct_train: training fraction of data; must be less than 1
    :param pct_dev: dev fraction of data; must be less than 1
    :param pct_test: testing fraction of data; must be less than 1
    :param target: Name of the target column; if specified, data will be stratified on this column

    :returns: A dictionary containing the keys train, test and dev with their respective data frames, as well as the "stratified_on" key indicating which columns the data was stratified on (None if it wasn't stratified on anything)
    """ # noqa

    if pct_train + pct_dev + pct_test != 1:
        raise Exception('The train, dev and test percentage of the data needs to sum up to 1')

    # Shuffle the data
    np.random.seed(seed)
    if not tss.is_timeseries:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split the data
    train_cutoff = round(data.shape[0] * pct_train)
    dev_cutoff = train_cutoff + round(data.shape[0] * pct_dev)

    train = data[:train_cutoff]
    dev = data[train_cutoff:dev_cutoff]
    test = data[dev_cutoff:]

    # Perform stratification if specified
    pcts = (pct_train, pct_dev, pct_test)
    train, dev, test, stratify_on = stratify_wrapper(train, dev, test, target, pcts, dtype_dict, tss)

    return {"train": train, "test": test, "dev": dev, "stratified_on": stratify_on}


def stratify_wrapper(train: pd.DataFrame,
                     dev: pd.DataFrame,
                     test: pd.DataFrame,
                     target: str,
                     pcts: (float, float, float),
                     dtype_dict: Dict[str, str],
                     tss: TimeseriesSettings) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, list):
    """
    Simple wrapper that acts as bridge between `splitter` and the actual stratification methods.

    :param train: train dataset
    :param dev: dev dataset
    :param test: test dataset
    :param target: Name of the target column; if specified, data will be stratified on this column
    :param pcts: tuple with (train, dev, test) fractions of the data
    :param dtype_dict: Dictionary with the data type of all columns
    :param tss: time-series specific details for splitting
    """
    stratify_on = []

    if target is not None:
        if dtype_dict[target] in (dtype.categorical, dtype.binary):
            stratify_on += [target]
        if tss.is_timeseries and isinstance(tss.group_by, list):
            stratify_on += tss.group_by

        if stratify_on:
            strat_fn = stratify if not tss.is_timeseries else ts_stratify
            train, dev, test = strat_fn(train, dev, test, pcts, stratify_on)

    return train, dev, test, stratify_on


def stratify(train: pd.DataFrame,
             dev: pd.DataFrame,
             test: pd.DataFrame,
             pcts: Tuple[float, float, float],
             stratify_on: List[str]) -> List[pd.DataFrame]:
    """
    Stratified data splitter.

    The `stratify_on` columns yield a cartesian product by which every different subset will be stratified
    independently from the others, and recombined at the end in fractions specified by `pcts`.

    For grouped time series tasks, stratification is done based on the group-by columns.

    :param train: Training data
    :param dev: Dev data
    :param test: Testing data
    :param pcts: tuple with (train, dev, test) fractions of the data
    :param stratify_on: Columns to consider when stratifying

    :returns Stratified train, dev, test dataframes
    """  # noqa

    data = pd.concat([train, dev, test])
    pct_train, pct_dev, pct_test = pcts
    train, dev, test = pd.DataFrame(columns=data.columns)

    all_group_combinations = list(product(*[data[col].unique() for col in stratify_on]))
    for group in all_group_combinations:
        subframe = data
        for idx, col in enumerate(stratify_on):
            subframe = subframe[subframe[col] == group[idx]]

        train_cutoff = round(subframe.shape[0] * pct_train)
        dev_cutoff = train_cutoff + round(subframe.shape[0] * pct_dev)

        train = train.append(subframe[:train_cutoff])
        dev = dev.append(subframe[train_cutoff:dev_cutoff])
        train = train.append(subframe[dev_cutoff:])

    return [train, dev, test]


def ts_stratify(train: pd.DataFrame,
                dev: pd.DataFrame,
                test: pd.DataFrame,
                pcts: Tuple[float, float, float],
                stratify_on: List[str]) -> List[pd.DataFrame]:
    """
    Stratified time series data splitter.
    
    The `stratify_on` columns yield a cartesian product by which every different subset will be stratified 
    independently from the others, and recombined at the end. 
    
    For grouped time series tasks, each group yields a different time series. That is, the splitter generates
    `nr_subsets` subsets from `data`, with equally-sized sub-series for each group.

    :param train: Training data
    :param dev: Dev data
    :param test: Testing data
    :param pcts: tuple with (train, dev, test) fractions of the data
    :param stratify_on: Columns to consider when stratifying

    :returns A list of data subsets, with each time series (as determined by `stratify_on`) equally split across them.
    """  # noqa
    data = pd.concat([train, dev, test])
    pct_train, pct_dev, pct_test = pcts
    gcd = np.gcd(100, np.gcd(pct_test, np.gcd(pct_train, pct_dev)))
    nr_subsets = int(100 / gcd)

    all_group_combinations = list(product(*[data[col].unique() for col in stratify_on]))

    subsets = [pd.DataFrame() for _ in range(nr_subsets)]
    for group in all_group_combinations:
        subframe = data
        for idx, col in enumerate(stratify_on):
            subframe = subframe[subframe[col] == group[idx]]

        subset = np.array_split(subframe, nr_subsets)

        for n in range(nr_subsets):
            subsets[n] = pd.concat([subsets[n], subset[n]])

    return subsets
