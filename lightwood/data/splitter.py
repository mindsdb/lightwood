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
    
    The proportion of data for each split must be specified (JSON-AI sets defaults to 80/10/10). First, rows in the dataset are shuffled randomly. Then a simple split is done. If a target value is provided and is of data type categorical/binary, then the splits will be stratified to maintain the representative populations of each class.

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

    if np.isclose(pct_train + pct_dev + pct_test, 1, atol=0.001) and np.less(pct_train + pct_dev + pct_test, 1+1e-5):
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
    Simple wrapper that determines whether stratification is needed, based on the target type and/or other parameters.

    :param train: Current train dataset, might be stratified depending on the criteria this method implements.
    :param dev: Current dev dataset, might be stratified depending on the criteria this method implements.
    :param test: Current test dataset, might be stratified depending on the criteria this method implements.
    :param target: Name of the target column; if specified, data will be stratified on this column.
    :param pcts: Tuple with (train, dev, test) fractions of the data.
    :param dtype_dict: Dictionary with the data type of all columns.
    :param tss: Time-series specific details for splitting.
    
    :returns Potentially stratified train, dev, test dataframes, along with a list of the columns by which the stratification was done.
    """  # noqa
    stratify_on = []

    if target is not None:
        if dtype_dict[target] in (dtype.categorical, dtype.binary):
            stratify_on += [target]
        if tss.is_timeseries and isinstance(tss.group_by, list):
            stratify_on += tss.group_by

        if stratify_on:
            train, dev, test = stratify(train, dev, test, pcts, stratify_on)

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
    train_st = pd.DataFrame(columns=data.columns)
    dev_st = pd.DataFrame(columns=data.columns)
    test_st = pd.DataFrame(columns=data.columns)

    all_group_combinations = list(product(*[data[col].unique() for col in stratify_on]))
    for group in all_group_combinations:
        df = data
        for idx, col in enumerate(stratify_on):
            df = df[df[col] == group[idx]]

        train_cutoff = round(df.shape[0] * pct_train)
        dev_cutoff = train_cutoff + round(df.shape[0] * pct_dev)

        train_st = train_st.append(df[:train_cutoff])
        dev_st = dev_st.append(df[train_cutoff:dev_cutoff])
        test_st = test_st.append(df[dev_cutoff:])

    return [train_st, dev_st, test_st]
