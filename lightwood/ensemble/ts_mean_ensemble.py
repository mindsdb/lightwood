from typing import List

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import LBFGS

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.mean_ensemble import MeanEnsemble
from lightwood.encoder.numeric.ts_array_numeric import TsArrayNumericEncoder
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.log import log


class TsMeanEnsemble(MeanEnsemble):
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict, horizon: int,
                 pred_args: PredictionArguments) -> None:
        super().__init__(target, mixers, data, dtype_dict)
        if not isinstance(data.encoders[target], TsArrayNumericEncoder):
            raise Exception('This ensemble can only be used to forecast!')

        self.horizon = horizon
        self.softmax = nn.Softmax(dim=0)
        # learn weights
        self.mixer_weights = nn.Parameter(torch.full((len(mixers),), 1 / len(mixers)))
        mixer_outputs = torch.tensor(self.predict(data, pred_args)).squeeze().reshape(-1, self.horizon, len(mixers))
        # Note: this assumes columns are previously sorted (which should always be true) TODO remove assumption
        target_cols = [target] + [c for c in data.data_frame.columns if f'{target}_timestep_' in c]
        actual = torch.tensor(data.data_frame[target_cols].values)
        nan_mask = actual != actual
        actual[nan_mask] = 0
        mixer_outputs[nan_mask] = 0
        criterion = nn.MSELoss()
        optimizer = LBFGS([self.mixer_weights], lr=0.01, max_iter=1000)

        def _eval_loss():
            optimizer.zero_grad()
            weighted = torch.sum(mixer_outputs * self.softmax(self.mixer_weights), dim=2)
            loss = criterion(weighted, actual)
            loss.backward()
            return loss

        optimizer.step(_eval_loss)
        log.info(f'Optimal stacking weights: {self.mixer_weights.detach().tolist()}')
        self.mixer_weights = torch.tensor([0, 1])  # self.softmax(self.mixer_weights)

    def predict(self, ds: EncodedDs, args: PredictionArguments) -> List:
        outputs = []
        for mixer in self.mixers:
            output = mixer(ds, args=args)['prediction'].tolist()
            output = np.expand_dims(np.array(output), 2)
            outputs.append(output)
        return outputs

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        output = pd.DataFrame()
        predictions = torch.tensor(self.predict(ds, args)).squeeze().reshape(-1, self.horizon, len(self.mixers))
        predictions = (predictions * self.mixer_weights).sum(axis=2)
        output['prediction'] = predictions.detach().numpy().tolist()
        return output
