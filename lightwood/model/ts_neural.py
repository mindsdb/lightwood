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
from lightwood.model.neural import Neural
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.model.helpers.default_net import DefaultNet
from lightwood.model.helpers.residual_net import ResidualNet
from lightwood.model.helpers.ar_net import ArNet
from lightwood.model.helpers.ranger import Ranger
from lightwood.model.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
from torch.optim.optimizer import Optimizer
from sklearn.metrics import r2_score


class TsNeural(Neural):
    model: nn.Module

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], input_cols: List[str], timeseries_settings: TimeseriesSettings, target_encoder: BaseEncoder):
        super().__init__(stop_after, target, dtype_dict, input_cols, timeseries_settings, target_encoder)

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        # ConcatedEncodedDs
        train_ds_arr = ds_arr[0:int(len(ds_arr) * 0.9)]
        test_ds_arr = ds_arr[int(len(ds_arr) * 0.9):]
        self.fit_data_len = len(ConcatedEncodedDs(train_ds_arr))

        self.model = ArNet(
            encoder_span=train_ds_arr[0].encoder_spans,
            target_name=self.target,
            input_size=len(ds_arr[0][0][0]),
            output_size=len(ds_arr[0][0][1])
        )
        
        criterion = self._select_criterion()
        started = time.time()
        scaler = GradScaler()

        full_test_dl = DataLoader(ConcatedEncodedDs(test_ds_arr), batch_size=200, shuffle=False)
        # Train on subsets
        best_full_test_error = pow(2, 32)
        best_model = None
        for subset_itt in (0, 1):
            for subset_idx in range(len(test_ds_arr)):
                train_dl = DataLoader(ConcatedEncodedDs(train_ds_arr[subset_idx * 9:(subset_idx + 1) * 9]), batch_size=200, shuffle=True)
                test_dl = DataLoader(test_ds_arr[subset_idx], batch_size=200, shuffle=False)

                # @TODO (Maybe) try adding wramup
                # Progressively decrease the learning rate
                total_epochs = 0
                running_errors: List[float] = []
                optimizer = self._select_optimizer(0.0005)
                for _ in range(int(20000)):
                    total_epochs += 1
                    error = self._run_epoch(train_dl, criterion, optimizer, scaler)
                    test_error = self._error(test_dl, criterion)
                    full_test_error = self._error(full_test_dl, criterion)
                    log.info(f'Training error of {error} | Testing error of {test_error} | During iteration {total_epochs} with subset {subset_idx}')
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
                        delta_mean = np.mean([running_errors[-i - 1] - running_errors[-i] for i in range(1, len(running_errors[-5:]))])
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
        self.partial_fit(test_ds_arr, train_ds_arr)
        self._final_tuning(test_ds_arr)
