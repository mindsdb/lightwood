from lightwood.encoder.base import BaseEncoder
from typing import Dict, List
import pandas as pd
from torch.nn.modules.loss import MSELoss
from lightwood.api import dtype
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
import time
from torch import nn
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.model.helpers.default_net import DefaultNet
from lightwood.model.helpers.residual_net import ResidualNet
from lightwood.model.helpers.ranger import Ranger
from lightwood.model.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from torch.optim.optimizer import Optimizer
from sklearn.metrics import r2_score


class Neural(BaseModel):
    model: nn.Module

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str], timeseries_settings: TimeseriesSettings, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.model = None
        self.dtype_dict = dtype_dict
        self.target = target
        self.timeseries_settings = timeseries_settings
        self.target_encoder = target_encoder
        self.epochs_to_best = 1
    
    def _final_tuning(self, data_arr):
        if self.dtype_dict[self.target] in (dtype.integer, dtype.float):
            self.model = self.model.eval()

            decoded_predictions = []
            deocded_real_values = []

            for data in data_arr:
                for X, Y in data:
                    X = X.to(self.model.device)
                    Y = Y.to(self.model.device)
                    Yh = self.model(X)

                    decoded_predictions.extend(self.target_encoder.decode(torch.unsqueeze(Yh, 0)))

                    deocded_real_values.extend(self.target_encoder.decode(torch.unsqueeze(Yh, 0)))

                self.target_encoder.decode_log = True
                log_acc = r2_score(deocded_real_values, decoded_predictions)
                self.target_encoder.decode_log = False
                lin_acc = r2_score(deocded_real_values, decoded_predictions)

                if lin_acc < log_acc:
                    self.target_encoder.decode_log = True
                else:
                    self.target_encoder.decode_log = False

    def _select_criterion(self) -> torch.nn.Module:
        if self.dtype_dict[self.target] in (dtype.categorical, dtype.binary):
            criterion = TransformCrossEntropyLoss(weight=self.target_encoder.index_weights.to(self.model.device))
        elif self.dtype_dict[self.target] in (dtype.tags):
            criterion = nn.BCEWithLogitsLoss()
        elif self.dtype_dict[self.target] in (dtype.integer, dtype.float) and self.timeseries_settings.is_timeseries:
            criterion = nn.L1Loss()
        elif self.dtype_dict[self.target] in (dtype.integer, dtype.float):
            criterion = MSELoss()
        else:
            criterion = MSELoss()

        return criterion

    def _select_optimizer(self, lr) -> Optimizer:
        if self.timeseries_settings.is_timeseries:
            optimizer = Ranger(self.model.parameters(), lr=lr)
        else:
            optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=2e-2)

        return optimizer
    
    def _run_epoch(self, train_dl, criterion, optimizer, scaler) -> float:
        self.model = self.model.train()
        running_losses: List[float] = []
        for X, Y in train_dl:
            X = X.to(self.model.device)
            Y = Y.to(self.model.device)
            with LightwoodAutocast():
                optimizer.zero_grad()
                Yh = self.model(X)
                loss = criterion(Yh, Y)
                if LightwoodAutocast.active:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            running_losses.append(loss.item())
        return np.mean(running_losses)
    
    def _error(self, test_dl, criterion) -> float:
        self.model = self.model.eval()
        running_losses: List[float] = []
        for X, Y in test_dl:
            X = X.to(self.model.device)
            Y = Y.to(self.model.device)
            Yh = self.model(X)
            running_losses.append(criterion(Yh, Y).item())
        return np.mean(running_losses)

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        # ConcatedEncodedDs
        train_ds_arr = ds_arr[0:int(len(ds_arr) * 0.9)]
        test_ds_arr = ds_arr[int(len(ds_arr) * 0.9):]

        self.model = DefaultNet(
            input_size=len(ds_arr[0][0][0]),
            output_size=len(ds_arr[0][0][1])
        )
        
        criterion = self._select_criterion()
        started = time.time()
        scaler = GradScaler()

        full_test_dl = DataLoader(ConcatedEncodedDs(test_ds_arr), batch_size=200, shuffle=False)
        # Train on subsets
        for subset_itt in (0, 1):
            for subset_idx in range(len(test_ds_arr)):
                train_dl = DataLoader(ConcatedEncodedDs(train_ds_arr[subset_idx * 9:(subset_idx + 1) * 9]), batch_size=200, shuffle=True)
                test_dl = DataLoader(test_ds_arr[subset_idx], batch_size=200, shuffle=False)

                best_model = deepcopy(self.model)
                best_full_test_error = pow(2, 32)

                # @TODO (Maybe) try adding wramup
                # Progressively decrease the learning rate
                total_epochs = 0
                running_errors: List[float] = []
                optimizer = self._select_optimizer(0.0005)
                for _ in range(int(1e10)):
                    total_epochs += 1
                    error = self._run_epoch(train_dl, criterion, optimizer, scaler)
                    test_error = self._error(test_dl, criterion)
                    full_test_error = self._error(full_test_dl, criterion)
                    log.info(f'Training error of {error} | Testing error of {test_error} | During iteration {total_epochs}')
                    running_errors.append(test_error)

                    if best_full_test_error > full_test_error:
                        best_full_test_error = full_test_error
                        best_model = deepcopy(self.model)
                        self.epochs_to_best = total_epochs

                    stop = False
                    if subset_itt == 0:
                        # Don't go through normal stopping logic, we don't want to assing the best model, this is just a "priming" iteration
                        break
                    elif len(running_errors) > 5:
                        delta_mean = np.mean([running_errors[-i] - running_errors[-i - 1] for i in range(1, len(running_errors))])
                        if delta_mean <= 0:
                            stop = True
                    elif np.isnan(error):
                        stop = True
                    elif (time.time() - started) > self.stop_after * (0.5 + subset_idx * 0.4 / len(test_ds_arr)):
                        stop = True
                    elif test_error < 0.00001:
                        stop = True

                    if stop:
                        self.model = best_model
                        break

        # Do a single training run on the test data as well
        self.partial_fit(test_ds_arr)
        self._final_tuning(test_ds_arr)
    
    def partial_fit(self, data: List[EncodedDs]) -> None:
        # Based this on how long the initial training loop took, at a low learning rate as to not mock anything up tooo badly
        ds = ConcatedEncodedDs(data)
        dl = DataLoader(ds, batch_size=200, shuffle=True)
        optimizer = self._select_optimizer(0.0005)
        criterion = self._select_criterion()
        scaler = GradScaler()
        for _ in range(max(1, int(self.epochs_to_best / 10))):
            self._run_epoch(dl, criterion, optimizer, scaler)

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        self.model = self.model.eval()
        decoded_predictions: List[object] = []
        
        dl = DataLoader
        for idx, (X, Y) in enumerate(ds):
            X = X.to(self.model.device)
            Y = Y.to(self.model.device)
            Yh = self.model(X)

            kwargs = {}
            for dep in self.target_encoder.dependencies:
                kwargs['dependency_data'] = {dep: ds.data_frame.iloc[idx][[dep]].values}
            decoded_prediction = self.target_encoder.decode(torch.unsqueeze(Yh, 0), **kwargs)
            decoded_predictions.extend(decoded_prediction)

        ydf = pd.DataFrame({'prediction': decoded_predictions})
        return ydf
