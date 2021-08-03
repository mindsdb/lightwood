from typing import List
from lightwood.model.base import BaseModel
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseEnsemble:
    data: List[EncodedDs]
    models: List[BaseModel]
    best_index: int
    supports_proba: bool

    def __init__(self, target, models: List[BaseModel], data: List[EncodedDs]) -> None:
        self.data = data
        self.models = models
        self.best_index = 0
        self.supports_proba = False
        
    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        raise NotImplementedError()
