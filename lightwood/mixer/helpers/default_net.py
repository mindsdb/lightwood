import math
import torch
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_device_from_name
from lightwood.helpers.log import log
import numpy as np


class DefaultNet(torch.nn.Module):
    """
    Default neural network module used in `Neural` mixer to learn the predictive task based on encoded feature representations.
    
    Small architecture with either a one or two linear layers. Hidden size might be constrained to have at most `max_params` in the network, trading off accuracy for speed.
    """  # noqa

    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 shape: list = None,
                 max_params: int = int(3e7),
                 num_hidden: int = 1,
                 dropout: float = 0,
                 device: str = '') -> None:

        super(DefaultNet, self).__init__()

        if input_size is not None and output_size is not None:
            self.input_size = input_size
            self.output_size = output_size
            hidden_size = max([self.input_size * 2, self.output_size * 2, 400])
            shape = [self.input_size] + [hidden_size] * num_hidden + [self.output_size]

            # If the network is too big, shrink it
            if np.sum([shape[i] * shape[i + 1] for i in range(len(shape) - 1)]) > max_params:
                log.warning('Shrinking network!')
                hidden_size = math.floor(max_params / (self.input_size * self.output_size))

                if hidden_size > self.output_size:
                    shape = [self.input_size, hidden_size, self.output_size]
                else:
                    shape = [self.input_size, self.output_size]
        if shape is not None:
            layers = []
            for ind in range(len(shape) - 1):
                layers.append(torch.nn.Linear(shape[ind], shape[ind + 1]))
                if ind < len(shape) - 2:
                    layers.append(torch.nn.SELU())
                    if dropout > 0.001:
                        layers.append(torch.nn.Dropout(p=dropout))
        else:
            raise Exception('You must specify other a shape or an input and output size when creating a DefaultNet!')

        self.net = torch.nn.Sequential(*layers)

        self.to(get_device_from_name(device))

    def to(self, device: torch.device) -> torch.nn.Module:
        if 'cuda' not in str(torch.device) == 0:
            log.warning(
                'Creating neural network on CPU, it will be significantly slower than using a GPU, consider using a GPU instead') # noqa
        self.net = self.net.to(device)

        self.device = device
        return self

    def forward(self, input):
        try:
            with LightwoodAutocast():
                output = self.net(input)
        except Exception:
            output = self.net(input)
        return output
