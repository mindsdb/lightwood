import pandas as pd
from lightwood.data.encoded_ds import EncodedDs


class BaseMixer:
    fit_data_len: int
    stable: bool

    def __init__(self, stop_after: int):
        self.stop_after = stop_after
        self.supports_proba = None

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        raise NotImplementedError()

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        raise NotImplementedError()

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass
