from lightwood import dtype
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from itertools import product
from lightwood.api.types import TimeseriesSettings


def stratify(data: pd.DataFrame, nr_subsets: int, stratify_on: List[str], seed: int = None) -> List[pd.DataFrame]:
    """
    Produces a stratified split on a list of columns into a number of subsets

    :param data: Data to be split
    :param nr_subsets: Number of subsets to create
    :param stratify_on: Columns to stratify on
    :param seed: seed for shuffling the dataframe, defaults to ``len(data)``

    :returns A list of equally-sized data subsets that can be concatenated by the full data. This preserves the group-by columns.
    """  # noqa
    all_value_combinations = list(product(*[data[col].unique() for col in stratify_on]))

    if return_indexes:
        indexes = []
    subsets = [pd.DataFrame() for _ in range(nr_subsets)]
    for unique_value_combination in all_value_combinations:
        subframe = data
        # Filter this down until we only have the relevant rows
        for idx, col in enumerate(stratify_on):
            subframe = subframe[subframe[col] == unique_value_combination[idx]]

        subset = np.array_split(subframe, nr_subsets)

        for i in range(nr_subsets):
            subsets[i] = pd.concat([subsets[i], subset[i]])

    return subsets