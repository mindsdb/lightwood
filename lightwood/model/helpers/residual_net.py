from logging import exception
import math
from typing import List
import torch
from torch import nn
from functools import reduce

from torch.nn.modules.activation import SELU
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.helpers.log import log


class ResidualModule(nn.Module):
    def __init__(
        self,
        input_size
    ) -> None:
        """Initialize self."""
        intermediate_size = max([input_size * 2, 400])
        super().__init__()
        self.normalization = nn.BatchNorm1d(input_size)
        self.linear_first = nn.Linear(input_size, intermediate_size)
        self.activation_first = nn.SELU()
        self.linear_second = nn.Linear(intermediate_size, input_size)
        self.activation_second = nn.SELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        x_input = x
        x = self.normalization(x)
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.linear_second(x)
        x = self.dropout(x)
        x = x_input + x
        return x


class ResidualNet(torch.nn.Module):
    def __init__(self, input_size: int = None, output_size: int = None, shape: List[int] = None, max_params: int = int(3e5)) -> None:
        super(ResidualNet, self).__init__()
        self.net = torch.nn.Sequential(*([ResidualModule(input_size) for _ in range(3)] + [nn.Linear(input_size, output_size)]))
        self.to(*get_devices())

    def to(self, device: torch.device, available_devices: int) -> torch.nn.Module:
        self.net = self.net.to(device)
        if available_devices > 1:
            self.dp_wrapper_net = torch.nn.DataParallel(self.net)
        else:
            self.dp_wrapper_net = self.net

        self.device = device
        self.available_devices = available_devices
        return self

    def forward(self, input):
        try:
            with LightwoodAutocast():
                output = self.net(input)
        except Exception as e:
            # Data parallel error
            if 'nccl' in str(e).lower():
                self.dp_wrapper_net = self.net
                log.warn(f'Data parallel not working: {e}')
            else:
                raise e

        return output
