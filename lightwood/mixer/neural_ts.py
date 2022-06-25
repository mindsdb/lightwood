from typing import Dict, Optional

import torch
from torch import nn
import torch_optimizer as ad_optim
from torch.optim.optimizer import Optimizer

from lightwood.api import dtype
from lightwood.encoder.base import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs
from lightwood.mixer.neural import Neural
from lightwood.mixer.helpers.ar_net import ArNet
from lightwood.mixer.helpers.default_net import DefaultNet
from lightwood.api.types import TimeseriesSettings


class NeuralTs(Neural):
    def __init__(
            self, stop_after: float, target: str, dtype_dict: Dict[str, str],
            timeseries_settings: TimeseriesSettings, target_encoder: BaseEncoder, net: str, fit_on_dev: bool,
            search_hyperparameters: bool, n_epochs: Optional[int] = None):
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
            n_epochs
        )
        self.timeseries_settings = timeseries_settings
        assert self.timeseries_settings.is_timeseries
        self.net_class = DefaultNet if net == 'DefaultNet' else ArNet
        self.stable = True

    def _select_criterion(self) -> torch.nn.Module:
        if self.dtype_dict[self.target] in (dtype.integer, dtype.float, dtype.num_tsarray, dtype.quantity):
            criterion = nn.L1Loss()
        else:
            criterion = super()._select_criterion()

        return criterion

    def _select_optimizer(self) -> Optimizer:
        optimizer = ad_optim.Ranger(self.model.parameters(), lr=self.lr)
        return optimizer

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        self._fit(train_data, dev_data)
