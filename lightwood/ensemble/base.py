from typing import List
from lightwood.mixer.base import BaseMixer
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseEnsemble:
    data: EncodedDs
    mixers: List[BaseMixer]
    best_index: int
    supports_proba: bool

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs) -> None:
        self.data = data
        self.mixers = mixers
        self.best_index = 0
        self.supports_proba = False

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        raise NotImplementedError()
