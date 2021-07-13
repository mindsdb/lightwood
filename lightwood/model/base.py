from typing import List
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseModel:
    fit_data_len: int

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def fit(self, data: List[EncodedDs]) -> None:
        raise NotImplementedError()

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        raise NotImplementedError()

    def partial_fit(self, train_data: List[EncodedDs], test_data: List[EncodedDs]) -> None:
        pass
