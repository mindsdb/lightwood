from typing import List, Union
import pandas as pd
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs


def encode(encoders: List[BaseEncoder], folds: List[pd.DataFrame], target: str) -> Union[EncodedDs, List[EncodedDs]]:
    if isinstance(folds, pd.DataFrame):
        return EncodedDs(encoders, folds, target)

    encoded_ds_arr: List[EncodedDs] = []
    for fold in folds:
        encoded_ds_arr.append(EncodedDs(encoders, fold, target))
    return encoded_ds_arr
