import time
from copy import deepcopy
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch_optimizer as ad_optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from lightwood.api import dtype
from lightwood.api.types import PredictionArguments
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from lightwood.mixer.neural import Neural
from lightwood.mixer.helpers.ar_net import ArNet
from lightwood.mixer.helpers.default_net import DefaultNet
from lightwood.mixer.helpers.ts import _apply_stl_on_training, _stl_transform, _stl_inverse_transform
from lightwood.api.types import TimeseriesSettings


class NeuralTs(Neural):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            timeseries_settings: TimeseriesSettings,
            target_encoder: BaseEncoder,
            net: str,
            fit_on_dev: bool,
            search_hyperparameters: bool,
            ts_analysis: Dict[str, Dict],
            n_epochs: Optional[int] = None,
            use_stl: Optional[bool] = False
    ):
        """
        Subclassed Neural mixer used for time series forecasting scenarios. 
        
        :param stop_after: How long the total fitting process should take
        :param target: Name of the target column
        :param dtype_dict: Data type dictionary
        :param timeseries_settings: TimeseriesSettings object for time-series tasks, refer to its documentation for available settings.
        :param target_encoder: Reference to the encoder used for the target
        :param net: The network type to use (`DeafultNet` or `ArNet`)
        :param fit_on_dev: If we should fit on the dev dataset
        :param search_hyperparameters: If the network should run a more through hyperparameter search (currently disabled)
        :param n_epochs: amount of epochs that the network will be trained for. Supersedes all other early stopping criteria if specified.
        """ # noqa
        super().__init__(
            stop_after,
            target,
            dtype_dict,
            target_encoder,
            net,
            fit_on_dev,
            search_hyperparameters,
            n_epochs,
        )
        self.timeseries_settings = timeseries_settings
        assert self.timeseries_settings.is_timeseries
        self.ts_analysis = ts_analysis
        self.net_class = DefaultNet if net == 'DefaultNet' else ArNet
        self.stable = True
        self.use_stl = use_stl

    def _select_criterion(self) -> torch.nn.Module:
        if self.dtype_dict[self.target] in (dtype.integer, dtype.float, dtype.num_tsarray, dtype.quantity):
            criterion = nn.L1Loss()
        else:
            criterion = super()._select_criterion()

        return criterion

    def _select_optimizer(self) -> Optimizer:
        optimizer = ad_optim.Ranger(self.model.parameters(), lr=self.lr)
        return optimizer

    def _fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        :param train_data: The network is fit/trained on this
        :param dev_data: Data used for early stopping and hyperparameter determination
        """  # noqa
        self.started = time.time()
        original_train = deepcopy(train_data.data_frame)
        original_dev = deepcopy(dev_data.data_frame)

        # Use STL blocks if available
        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            _apply_stl_on_training(train_data, dev_data, self.target, self.timeseries_settings, self.ts_analysis)

        # ConcatedEncodedDs
        self.batch_size = min(200, int(len(train_data) / 10))
        self.batch_size = max(40, self.batch_size)

        dev_dl = DataLoader(dev_data, batch_size=self.batch_size, shuffle=False)
        train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        self.lr = 1e-4
        self.num_hidden = 1

        # Find learning rate
        # keep the weights
        self._init_net(train_data)
        self.lr, self.model = self._find_lr(train_dl)

        # Keep on training
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = GradScaler()

        # Only 0.8 of the remaining time budget is used to allow some time for the final tuning and partial fit
        self.model, epoch_to_best_model, _ = self._max_fit(
            train_dl, dev_dl, criterion, optimizer, scaler, (self.stop_after - (time.time() - self.started)) * 0.8,
            return_model_after=20000)

        self.epochs_to_best += epoch_to_best_model

        # restore dfs
        train_data.data_frame = original_train
        dev_data.data_frame = original_dev

        if self.fit_on_dev:
            self.partial_fit(dev_data, train_data)

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        self._fit(train_data, dev_data)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()
                 ) -> pd.DataFrame:
        original_df = deepcopy(ds.data_frame)

        self.model = self.model.eval()
        decoded_predictions = []
        all_probs: List[List[float]] = []
        rev_map = {}

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        pred_cols = [f'prediction_{i}' for i in range(self.timeseries_settings.horizon)]
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           dtype=object,
                           columns=pred_cols)

        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            ds.data_frame = _stl_transform(ydf, ds, self.target, self.timeseries_settings, self.ts_analysis)

        with torch.no_grad():
            for idx, (X, Y) in enumerate(ds):
                X = X.to(self.model.device)
                Yh = self.model(X)
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

        decoded_predictions = np.array(decoded_predictions)
        if len(decoded_predictions.shape) == 1:
            decoded_predictions = np.expand_dims(decoded_predictions, axis=1)
        ydf[pred_cols] = decoded_predictions

        if self.use_stl and self.ts_analysis.get('stl_transforms', False):
            ydf = _stl_inverse_transform(ydf, ds, self.timeseries_settings, self.ts_analysis)

        ydf['prediction'] = ydf.values.tolist()

        if self.timeseries_settings.horizon == 1:
            ydf['prediction'] = [p[0] for p in ydf['prediction']]

        if args.predict_proba and self.supports_proba:
            raw_predictions = np.array(all_probs).squeeze(axis=1)

            for idx, label in enumerate(rev_map.values()):
                ydf[f'__mdb_proba_{label}'] = raw_predictions[:, idx]

        # TODO: make this part of the base mixer class? to avoid repetitive code
        #  and ensure other contribs don't accidentally modify the df
        ds.data_frame = original_df
        return ydf[['prediction']]
