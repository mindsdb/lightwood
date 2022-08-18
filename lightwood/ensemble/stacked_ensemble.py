from typing import List, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import SGD

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.mean_ensemble import MeanEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.log import log


class StackedEnsemble(MeanEnsemble):
    """
    This ensemble will learn an optimal weight vector via Stochastic Gradient Descent on the validation dataset and the respective mixer predictions.

    Starting weights for the vector are uniformly set.

    Note this mixer is still in experimental phase. Some features in the roadmap are:
      - support for handling faulty mixers
      - support for custom initial vector weights
      - early stopping
      - arbitrarily complex secondary model

    """  # noqa
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict,
                 args: PredictionArguments, fit: Optional[bool] = True, **kwargs) -> None:
        super().__init__(target, mixers, data, dtype_dict, fit=False)

        self.target_cols = [target]
        self.softmax = nn.Softmax(dim=0)
        self.mixer_weights = nn.Parameter(torch.full((len(mixers),), 1 / len(mixers)))
        self.criterion = nn.MSELoss()
        self.optimizer = SGD([self.mixer_weights], lr=0.01)
        self.agg_dim = 1

        if fit:
            all_preds = torch.tensor(self.predict(data, args)).squeeze().reshape(-1, len(mixers))
            actual = torch.tensor(data.data_frame[self.target_cols].values)

            def _eval_loss():
                self.optimizer.zero_grad()
                weighted = torch.sum(all_preds * self.softmax(self.mixer_weights), dim=self.agg_dim)
                loss = self.criterion(weighted, actual)
                loss.backward()
                return loss

            self.optimizer.step(_eval_loss)
            self.mixer_weights = self.softmax(self.mixer_weights)
            log.info(f'Optimal stacking weights: {self.mixer_weights.detach().tolist()}')
            self.prepared = True

    def predict(self, ds: EncodedDs, args: PredictionArguments) -> List:
        outputs = []
        for mixer in self.mixers:
            output = mixer(ds, args=args)['prediction'].tolist()
            output = np.expand_dims(np.array(output), self.agg_dim)
            outputs.append(output)
        return outputs

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        assert self.prepared
        output = pd.DataFrame()
        predictions = torch.tensor(self.predict(ds, args)).squeeze().reshape(-1, len(self.mixers))
        predictions = (predictions * self.mixer_weights).sum(axis=1)
        output['prediction'] = predictions.detach().numpy().tolist()
        return output

    def set_weights(self, weights: List):
        if len(weights) != len(self.mixers):
            raise Exception(f"Expected weight vector to have {len(self.mixers)} entries, got {len(weights)} instead.")

        self.mixer_weights = torch.tensor(weights)
