from typing import List
import pandas as pd
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs

InputData = Union[pd.DataFrame, List[pd.DataFrame]]

def encode(encoders: List[BaseEncoder], subsets: InputData, target: str) -> List[EncodedDs]:
    """
    Given a list of Lightwood encoders, and data subsets, applies the encoders onto each subset.

    :param encoders: A list of lightwood encoders, in the order of each of the column types.
    :param folds: A list of data subsets, each being a separate dataframe with all the columns applied per encoder.
    :param target: The name of the column that is the target for prediction.

    :returns: An encoded dataset for each encoder in the list
    """
    if isinstance(subsets, pd.DataFrame):
        subsets = [subsets]

    encoded_ds_arr: List[EncodedDs] = []
    for subset in subsets:
        encoded_ds_arr.append(EncodedDs(encoders, subset, target))
    return encoded_ds_arr
