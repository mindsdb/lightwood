from typing import List
import pandas as pd
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs


def encode(encoders: List[BaseEncoder], subsets: List[pd.DataFrame], target: str) -> List[EncodedDs]:
    if isinstance(subsets, pd.DataFrame):
        subsets = [subsets]

    encoded_ds_arr: List[EncodedDs] = []
    for fold in subsets:
        encoded_ds_arr.append(EncodedDs(encoders, fold, target))
    return encoded_ds_arr
