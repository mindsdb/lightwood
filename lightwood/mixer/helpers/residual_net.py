from typing import List
import torch
from torch import nn
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        x_input = x
        if self.training:
            x = self.normalization(x)
        x = self.linear_first(x)
        x = self.activation_first(x)
        x = self.dropout(x)
        x = self.linear_second(x)
        x = x_input + x
        return x


class ResidualNet(torch.nn.Module):
    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 shape: List[int] = None,
                 max_params: int = int(3e5)) -> None:
        super(ResidualNet, self).__init__()
        self.net = torch.nn.Sequential(
            *
            ([ResidualModule(input_size) for _ in range(1)] +
             [nn.Linear(input_size, max([input_size * 2, output_size * 2, 400])),
              nn.Linear(max([input_size * 2, output_size * 2, 400]),
                        output_size)]))
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
