import time
from copy import deepcopy
from typing import Dict, List, Optional

import torch
# import torch.nn as nn
import numpy as np
import pandas as pd
from tab_transformer_pytorch import FTTransformer

# from type_infer.dtype import dtype
from lightwood.helpers.log import log
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.api.types import PredictionArguments
from lightwood.helpers.device import get_device_from_name
from lightwood.data.encoded_ds import EncodedDs
from lightwood.encoder.base import BaseEncoder
from lightwood.mixer.neural import Neural


class TabTransformerMixer(Neural):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            target_encoder: BaseEncoder,
            fit_on_dev: bool,
            search_hyperparameters: bool,
            train_args: Optional[dict] = None
    ):
        """
        This mixer trains a TabTransformer network (FT variant), using concatenated encoder outputs for each dataset feature as input, to predict the encoded target column representation as output.
        
        Training logic is based on the Neural mixer, please refer to it for more details on each input parameter.
        """  # noqa
        self.train_args = train_args if train_args else {}
        super().__init__(
            stop_after,
            target,
            dtype_dict,
            target_encoder,
            'FTTransformer',
            False,  # fit_on_dev
            search_hyperparameters,
            n_epochs=self.train_args.get('n_epochs', None)
        )
        self.lr = self.train_args.get('lr')
        self.stable = True  # still experimental

    def _init_net(self, ds: EncodedDs):
        self.net_class = FTTransformer

        self.model = FTTransformer(
            categories=(),                                                       # unused here, as by the point it arrives to the mixer, everything is numerical  # noqa
            num_continuous=len(ds[0][0]),  # ds.input_length,                         # TODO define based on DS
            dim=self.train_args.get('dim', 32),
            dim_out=self.train_args.get('dim_out', len(ds[0][1])),
            depth=self.train_args.get('depth', 6),
            heads=self.train_args.get('heads', 8),
            attn_dropout=self.train_args.get('attn_dropout', 0.1),               # post-attention dropout
            ff_dropout=self.train_args.get('ff_dropout', 0.1),                   # feed forward dropout
            mlp_hidden_mults=self.train_args.get('mlp_hidden_mults', (4, 2)),    # relative multiples of each hidden dimension of the last mlp to logits  # noqa
            # mlp_act=self.train_args.get('mlp_act', nn.ReLU()),  # TODO: import string from nn activations
        )
        self.model.device = get_device_from_name('')
        self.model.to(self.model.device)

    def _max_fit(self, train_dl, dev_dl, criterion, optimizer, scaler, stop_after, return_model_after):
        epochs_to_best = 0
        best_dev_error = pow(2, 32)
        running_errors = []
        best_model = self.model

        for epoch in range(1, return_model_after + 1):
            self.model = self.model.train()
            running_losses: List[float] = []
            for i, (X, Y) in enumerate(train_dl):
                X = X.to(self.model.device)
                Y = Y.to(self.model.device)
                with LightwoodAutocast():
                    optimizer.zero_grad()
                    Yh = self.model(torch.Tensor(), X)
                    loss = criterion(Yh, Y)
                    if LightwoodAutocast.active:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                running_losses.append(loss.item())
                if (time.time() - self.started) > stop_after:
                    break

            train_error = np.mean(running_losses)
            epoch_error = self._error(dev_dl, criterion)
            running_errors.append(epoch_error)
            log.info(f'Loss @ epoch {epoch}: {epoch_error}')

            if np.isnan(train_error) or np.isnan(
                    running_errors[-1]) or np.isinf(train_error) or np.isinf(
                    running_errors[-1]):
                break

            if best_dev_error > running_errors[-1]:
                best_dev_error = running_errors[-1]
                best_model = deepcopy(self.model)
                epochs_to_best = epoch

            # manually set epoch limit
            if self.n_epochs is not None:
                if epoch > self.n_epochs:
                    break

            # automated early stopping
            else:
                if len(running_errors) >= 5:
                    delta_mean = np.average([running_errors[-i - 1] - running_errors[-i] for i in range(1, 5)],
                                            weights=[(1 / 2)**i for i in range(1, 5)])
                    if delta_mean <= 0:
                        break
                elif (time.time() - self.started) > stop_after:
                    break
                elif running_errors[-1] < 0.0001 or train_error < 0.0001:
                    break

        if np.isnan(best_dev_error):
            best_dev_error = pow(2, 32)
        return best_model, epochs_to_best, best_dev_error

    def _error(self, dev_dl, criterion) -> float:
        self.model = self.model.eval()
        running_losses: List[float] = []
        with torch.no_grad():
            for X, Y in dev_dl:
                X = X.to(self.model.device)
                Y = Y.to(self.model.device)
                Yh = self.model(torch.Tensor(), X)
                running_losses.append(criterion(Yh, Y).item())
            return np.mean(running_losses)

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        self._fit(train_data, dev_data)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Make predictions based on datasource with the same features as the ones used for fitting

        :param ds: Predictions are generate from it
        :param arg: Any additional arguments used in predicting

        :returns: A dataframe cotaining the decoded predictions and (depending on the args) additional information such as the probabilites for each target class
        """ # noqa
        self.model = self.model.eval()
        decoded_predictions: List[object] = []
        all_probs: List[List[float]] = []
        rev_map = {}

        with torch.no_grad():
            for idx, (X, Y) in enumerate(ds):
                X = X.to(self.model.device)
                Yh = self.model(torch.Tensor(), X.unsqueeze(0))
                Yh = torch.unsqueeze(Yh, 0) if len(Yh.shape) < 2 else Yh

                kwargs = {}
                for dep in self.target_encoder.dependencies:
                    kwargs['dependency_data'] = {dep: ds.data_frame.iloc[idx][[dep]].values}

                if args.predict_proba and self.supports_proba:
                    decoded_prediction, probs, rev_map = self.target_encoder.decode_probabilities(Yh, **kwargs)
                    all_probs.append(probs)
                else:
                    decoded_prediction = self.target_encoder.decode(Yh, **kwargs)

                decoded_predictions.extend(decoded_prediction)

            ydf = pd.DataFrame({'prediction': decoded_predictions})

            if args.predict_proba and self.supports_proba:
                raw_predictions = np.array(all_probs).squeeze(axis=1)

                for idx, label in enumerate(rev_map.values()):
                    ydf[f'__mdb_proba_{label}'] = raw_predictions[:, idx]

            return ydf