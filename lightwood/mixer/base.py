import pandas as pd

from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments


class BaseMixer:
    fit_data_len: int
    stable: bool
    supports_proba: bool

    def __init__(self, stop_after: int):
        self.stop_after = stop_after
        self.supports_proba = False

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        raise NotImplementedError()

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        raise NotImplementedError()

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass
