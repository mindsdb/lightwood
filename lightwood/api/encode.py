from typing import List
import pandas as pd
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs

def encode(encoders: List[BaseEncoder], folds: List[pd.DataFrame], target: str) -> List[EncodedDs]:
    encoded_ds_arr: List[EncodedDs] = []
    for fold in folds:
        encoded_ds_arr.append(EncodedDs(encoders, fold, target))
    return encoded_ds_arr

def concat(encoded_ds_arr: List[EncodedDs]):
    return ConcatedEncodedDs(encoded_ds_arr)