from typing import List
import pandas as pd
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs

def encode(encoders: List[BaseEncoder], folds: List[pd.DataFrame], target: str) -> List[EncodedDs]:
    encoded_ds_arr