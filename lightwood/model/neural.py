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
import torch_optimizer as ad_optim
from lightwood.model.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from torch.optim.optimizer import Optimizer
from sklearn.metrics import r2_score
import optuna


class Neural(BaseModel):
    model: nn.Module

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str], timeseries_settings: TimeseriesSettings, target_encoder: BaseEncoder, fit_on_dev: bool):
        super().__init__(stop_after)
        self.model = None
        self.dtype_dict = dtype_dict
        self.target = target
        self.timeseries_settings = timeseries_settings
        self.target_encoder = target_encoder
        self.epochs_to_best = 0
        self.fit_on_dev = fit_on_dev
    
    def _final_tuning(self, data_arr):
        if self.dtype_dict[self.target] in (dtype.integer, dtype.float):
            self.model = self.model.eval()

            decoded_predictions = []
            decoded_real_values = []

            for data in data_arr:
                for X, Y in data:
                    X = X.to(self.model.device)
                    Y = Y.to(self.model.device)
                    Yh = self.model(X)

                    decoded_predictions.extend(self.target_encoder.decode(torch.unsqueeze(Yh, 0)))

                    decoded_real_values.extend(self.target_encoder.decode(torch.unsqueeze(Y, 0)))

                self.target_encoder.decode_log = True
                log_acc = r2_score(decoded_real_values, decoded_predictions)
                self.target_encoder.decode_log = False
                lin_acc = r2_score(decoded_real_values, decoded_predictions)

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
            optimizer = ad_optim.Ranger(self.model.parameters(), lr=lr)
        else:
            optimizer = ad_optim.Ranger(self.model.parameters(), lr=lr, weight_decay=2e-2)

        return optimizer

    def _run_epoch(self, train_dl, criterion, optimizer, scaler, scheduler) -> float:
        self.model = self.model.train()
        running_losses: List[float] = []
        for i, (X, Y) in enumerate(train_dl):
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

            if i % 4 == 3 and scheduler is not None:
                scheduler.step()

        return np.mean(running_losses)

    def _max_fit(self, train_dl, dev_dl, criterion, optimizer, scheduler, scaler, stop_after, return_model_after):
        started = time.time()
        epochs_to_best = 0
        best_dev_error = pow(2, 32)
        running_errors = []
        best_model = self.model

        train_error = None
        for epoch in range(1, return_model_after + 1):
            train_error = self._run_epoch(train_dl, criterion, optimizer, scaler, scheduler)
            running_errors.append(self._error(dev_dl, criterion))

            if np.isnan(train_error) or np.isnan(running_errors[-1]) or np.isinf(train_error) or np.isinf(running_errors[-1]):
                break

            if best_dev_error > running_errors[-1]:
                best_dev_error = running_errors[-1]
                best_model = deepcopy(self.model)
                epochs_to_best = epoch

            if len(running_errors) > 15:
                delta_mean = np.mean([running_errors[-i - 1] - running_errors[-i] for i in range(1, len(running_errors[-7:]))])
                if delta_mean <= 0:
                    break
            elif (time.time() - started) > stop_after:
                break
            elif running_errors[-1] < 0.0001 or train_error < 0.0001:
                break

        if np.isnan(best_dev_error):
            best_dev_error = pow(2, 32)
        return best_model, epochs_to_best, best_dev_error

    def _error(self, dev_dl, criterion) -> float:
        self.model = self.model.eval()
        running_losses: List[float] = []
        for X, Y in dev_dl:
            X = X.to(self.model.device)
            Y = Y.to(self.model.device)
            Yh = self.model(X)
            running_losses.append(criterion(Yh, Y).item())
        return np.mean(running_losses)

    # @TODO: Compare partial fitting fully on and fully off on the benchmarks!
    # @TODO: Writeup on the methodology for partial fitting
    def fit(self, ds_arr: List[EncodedDs]) -> None:
        # ConcatedEncodedDs
        train_ds_arr = ds_arr[0:int(len(ds_arr) * 0.9)]
        dev_ds_arr = ds_arr[int(len(ds_arr) * 0.9):]

        scaler = GradScaler()
        self.batch_size = min(200, int(len(ConcatedEncodedDs(ds_arr)) / 20))

        time_for_trials = self.stop_after / 2
        nr_trails = 30
        time_per_trial = time_for_trials / nr_trails

        def objective(trial):
            log.debug(f'Running trial in max {time_per_trial} seconds')
            # For trail options see: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest_int
            num_hidden = trial.suggest_int('num_hidden', 1, 2)
            lr = trial.suggest_loguniform('lr', 0.0003, 0.05)
            dropout = trial.suggest_uniform('dropout', 0, 0.35)

            self.model = DefaultNet(
                input_size=len(ds_arr[0][0][0]),
                output_size=len(ds_arr[0][0][1]),
                num_hidden=num_hidden,
                dropout=dropout
            )
            optimizer = self._select_optimizer(lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0.0005, last_epoch=-1)
            criterion = self._select_criterion()
            
            dev_dl = DataLoader(ConcatedEncodedDs(train_ds_arr[0:int(len(train_ds_arr) * 0.7)]), batch_size=self.batch_size, shuffle=False)
            train_dl = DataLoader(ConcatedEncodedDs(train_ds_arr[int(len(train_ds_arr) * 0.7):]), batch_size=self.batch_size, shuffle=False)
            try:
                _, _, best_error = self._max_fit(train_dl, dev_dl, criterion, optimizer, scheduler, scaler, time_per_trial, 20000)
            except Exception as e:
                log.error(e)
                return pow(2, 32)

            return best_error

        log.info('Running hyperparameter search!')
        sampler = optuna.samplers.RandomSampler(seed=len(ds_arr[0][0][0]))
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=nr_trails)

        log.debug(f'Best trial had a loss of : {study.best_trial.value}')
        log.debug(f'Best trial suggested parameters : {study.best_trial.params.items()}')

        self.num_hidden = study.best_trial.params['num_hidden']
        self.dropout = study.best_trial.params['dropout']
        self.lr = study.best_trial.params['lr']
        dev_dl = DataLoader(ConcatedEncodedDs(dev_ds_arr), batch_size=self.batch_size, shuffle=False)
        train_dl = DataLoader(ConcatedEncodedDs(train_ds_arr), batch_size=self.batch_size, shuffle=False)

        log.info(f'Found hyperparameters num_hidden:{self.num_hidden} lr:{self.lr} dropout:{self.dropout}')

        self.model = DefaultNet(
            input_size=len(ds_arr[0][0][0]),
            output_size=len(ds_arr[0][0][1]),
            num_hidden=self.num_hidden,
            dropout=self.dropout
        )
        optimizer = self._select_optimizer(self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0.0005, last_epoch=-1)
        criterion = self._select_criterion()

        self.model, self.epochs_to_best, best_error = self._max_fit(train_dl, dev_dl, criterion, optimizer, scheduler, scaler, self.stop_after / 2, 20000)
        '''
        for subset_itt in (0, 1):
            for subset_idx in range(len(dev_ds_arr)):
                train_dl = DataLoader(ConcatedEncodedDs(train_ds_arr[subset_idx * 9:(subset_idx + 1) * 9]), batch_size=200, shuffle=True)

                optimizer = self._select_optimizer(0.005)
                stop_after = (self.stop_after / 2) * (0.5 + subset_idx * 0.4 / len(dev_ds_arr))
                self.model, epoch_to_best_model = self._max_fit(train_dl, dev_dl, criterion, optimizer, scaler, stop_after, return_model_after=20000 if subset_itt > 0 else 1)
                self.epochs_to_best += epoch_to_best_model
        '''

        # Do a single training run on the test data as well
        if self.fit_on_dev:
            self.partial_fit(dev_ds_arr, train_ds_arr)
        self._final_tuning(dev_ds_arr)
    
    def partial_fit(self, train_data: List[EncodedDs], dev_data: List[EncodedDs]) -> None:
        # Based this on how long the initial training loop took, at a low learning rate as to not mock anything up tooo badly
        train_ds = ConcatedEncodedDs(train_data)
        dev_ds = ConcatedEncodedDs(dev_data + train_data)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dev_dl = DataLoader(dev_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = self._select_optimizer(self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0.0005, last_epoch=-1)
        criterion = self._select_criterion()
        scaler = GradScaler()

        self.model, _, _ = self._max_fit(train_dl, dev_dl, criterion, optimizer, scheduler, scaler, self.stop_after, return_model_after=self.epochs_to_best)
    
    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        self.model = self.model.eval()
        decoded_predictions: List[object] = []
        
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
