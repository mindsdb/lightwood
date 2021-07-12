from typing import List
from lightwood.model.base import BaseModel
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseEnsemble:
    models: List[BaseModel]

    def __init__(self, models: List[BaseModel]) -> None:
        self.models = models
        
    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        raise NotImplementedError()
