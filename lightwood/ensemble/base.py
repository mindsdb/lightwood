from typing import List
from lightwood.model.base import BaseModel
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseEnsemble:
    data: List[EncodedDs]
    models: List[BaseModel]

    def __init__(self, models: List[BaseModel], data: List[EncodedDs]) -> None:
        self.data = data
        self.models = models
        
    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        raise NotImplementedError()
