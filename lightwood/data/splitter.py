import pandas as pd
import numpy as np
from typing import List


def splitter(data: pd.DataFrame, k: int) -> List[pd.DataFrame]:
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    # split
    folds = np.array_split(data, k)
    return folds
