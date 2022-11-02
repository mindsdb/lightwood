from copy import deepcopy
from typing import List, Optional

import torch
from torch import nn
from torch.optim import SGD
import numpy as np
import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.stacked_ensemble import StackedEnsemble
from lightwood.encoder.array.ts_num_array import TsArrayNumericEncoder
from lightwood.api.types import PredictionArguments
from type_infer.dtype import dtype
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.log import log


class TsStackedEnsemble(StackedEnsemble):
    """
    Thin wrapper for `StackedEnsemble` that enables forecasting support.
    """
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict, ts_analysis: dict,
                 args: PredictionArguments, fit: Optional[bool] = True, **kwargs) -> None:
        dtype_dict = deepcopy(dtype_dict)
        dtype_dict[target] = dtype.float  # hijack to correctly initialize parent class
        super().__init__(target, mixers, data, dtype_dict, args, fit=False)
        if not isinstance(data.encoders[target], TsArrayNumericEncoder):
            raise Exception('This ensemble can only be used to forecast!')
        self.ts_analysis = ts_analysis
        self.horizon = self.ts_analysis['tss'].horizon
        self.target_cols = [target] + [f'{target}_timestep_{t+1}' for t in range(self.horizon - 1)]
        self.agg_dim = 2
        self.opt_max_iter = 1000

        if fit:
            all_preds = torch.tensor(self.predict(data, args)).squeeze().reshape(-1, self.horizon, len(mixers))
            actual = torch.tensor(data.data_frame[self.target_cols].values)
            nan_mask = actual != actual
            actual[nan_mask] = 0
            all_preds[nan_mask, :] = 0

            criterion = nn.SmoothL1Loss()
            optimizer = SGD([self.mixer_weights], lr=1e-3)

            def _eval_loss():
                optimizer.zero_grad()
                weighted = torch.sum(all_preds * self.softmax(self.mixer_weights), dim=self.agg_dim)
                loss = criterion(weighted, actual)
                loss.backward()
                return loss

            for _ in range(self.opt_max_iter):
                optimizer.step(_eval_loss)
            self.mixer_weights = self.softmax(self.mixer_weights)
            log.info(f'Optimal stacking weights: {self.mixer_weights.detach().tolist()}')
            self.prepared = True

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        assert self.prepared
        output = pd.DataFrame()
        predictions = torch.tensor(np.concatenate(self.predict(ds, args), axis=2))
        predictions = (predictions * self.mixer_weights).sum(axis=self.agg_dim)
        output['prediction'] = predictions.detach().numpy().tolist()
        return output
