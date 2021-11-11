"""
2021.07.16

For encoders that already fine-tune on the targets (namely text)
the unity mixer just arg-maxes the output of the encoder.
"""

from typing import List

import torch
import pandas as pd

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments


class Unit(BaseMixer):
    def __init__(self, stop_after: float, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.supports_proba = False
        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info("Unit Mixer just borrows from encoder")

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        if args.predict_proba:
            # @TODO: depending on the target encoder, this might be enabled
            log.warning('This model does not output probability estimates')

        decoded_predictions: List[object] = []

        for X, _ in ds:
            decoded_prediction = self.target_encoder.decode(torch.unsqueeze(X, 0))
            decoded_predictions.extend(decoded_prediction)

        ydf = pd.DataFrame({"prediction": decoded_predictions})
        return ydf
