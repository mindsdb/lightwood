from typing import List, Optional

import torch
import pandas as pd

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments


class Unit(BaseMixer):
    def __init__(self, stop_after: float, target_encoder: BaseEncoder):
        """
        The "Unit" mixer serves as a simple wrapper around a target encoder, essentially borrowing 
        the encoder's functionality for predictions. In other words, it simply arg-maxes the output of the encoder

        Used with encoders that already fine-tune on the targets (namely, pre-trained text ML models).
        
        Attributes:
            :param target_encoder: An instance of a Lightwood BaseEncoder. This encoder is used to decode predictions.
            :param stop_after (float): Time budget (in seconds) to train this mixer. 
        """  # noqa
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.supports_proba = False
        self.stable = True

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        log.info("Unit mixer does not require training, it passes through predictions from its encoders.")

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs, args: Optional[dict] = None) -> None:
        pass

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Makes predictions using the provided EncodedDs dataset.
        Mixer decodes predictions using the target encoder and returns them in a pandas DataFrame.

        :returns ydf (pd.DataFrame): a data frame containing the decoded predictions.
        """
        if args.predict_proba:
            # @TODO: depending on the target encoder, this might be enabled
            log.warning('This model does not output probability estimates')

        decoded_predictions: List[object] = []

        for X, _ in ds:
            decoded_prediction = self.target_encoder.decode(torch.unsqueeze(X, 0))
            decoded_predictions.extend(decoded_prediction)

        ydf = pd.DataFrame({"prediction": decoded_predictions})
        return ydf
