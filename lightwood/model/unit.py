"""
2021.07.16

For encoders that already fine-tune on the targets (namely text)
the unity mixer just arg-maxes the output of the encoder.
"""

from typing import List
from lightwood.encoder.base import BaseEncoder
from lightwood.model.base import BaseModel
from lightwood.helpers.log import log
from lightwood.data.encoded_ds import EncodedDs
import pandas as pd
import torch


class Unit(BaseModel):
    def __init__(self, stop_after: int, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.target_encoder = target_encoder

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info("Unit Mixer just borrows from encoder")

    def partial_fit(
        self, train_data: List[EncodedDs], dev_data: List[EncodedDs]
    ) -> None:
        pass

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:

        decoded_predictions: List[object] = []

        for X, _ in ds:
            decoded_prediction = self.target_encoder.decode(torch.unsqueeze(X, 0))
            decoded_predictions.extend(decoded_prediction)

        ydf = pd.DataFrame({"prediction": decoded_predictions})
        return ydf
