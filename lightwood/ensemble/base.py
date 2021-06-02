from typing import List
from lightwood.model.base import BaseModel
from lightwood.api.types import LightwoodConfig
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseEnsemble:
    lightwood_config: LightwoodConfig
    data: EncodedDs
    models: List[BaseModel]

    def __init__(self, models: List[BaseModel], data: EncodedDs, lightwood_config: LightwoodConfig) -> None:
        self.data = data
        self.lightwood_config = lightwood_config
        self.models = models
        
    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        raise NotImplementedError()
