from typing import List

import torch
import pandas as pd
from sklearn.linear_model import LinearRegression

from lightwood.helpers.log import log
from lightwood.model import BaseModel
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs


class Regression(BaseModel):
    model: LinearRegression
    
    def __init__(self, stop_after: int, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.supports_proba = False

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting Regression model')
        X = []
        Y = []
        for x, y in ConcatedEncodedDs(ds_arr):
            X.append(x.tolist())
            Y.append(y.tolist())

        self.model = LinearRegression().fit(X, Y)
        log.info(f'Regression based correlation of: {self.model.score(X, Y)}')

    def partial_fit(self, train_data: List[EncodedDs], dev_data: List[EncodedDs]) -> None:
        self.fit(train_data + dev_data)

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        if predict_proba:
            log.warning('This model cannot output probability estimates')
        X = []
        for x, _ in ds:
            X.append(x.tolist())
        
        Yh = self.model.predict(X)

        decoded_predictions = self.target_encoder.decode(torch.Tensor(Yh))

        ydf = pd.DataFrame({'prediction': decoded_predictions})
        return ydf
