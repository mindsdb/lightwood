from typing import List
import pandas as pd
from lightwood.encoders.base import BaseEncoder


def encode(encoders: List[BaseEncoder], folds: List[pd.DataFrame]) -> List[pd.DataFrame]:
    return folds
