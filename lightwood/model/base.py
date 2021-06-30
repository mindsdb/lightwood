from lightwood.api.types import JsonML
from typing import List
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseModel:
    json_ml: JsonML

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def fit(self, data: List[EncodedDs]) -> None:
        raise NotImplementedError()

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        raise NotImplementedError()

    def partial_fit(self, data: List[EncodedDs]) -> None:
        pass
