import math
from typing import List
import torch
from functools import reduce
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices


class DefaultNet(torch.nn.Module):
    def __init__(self, input_size: int = None, output_size: int = None, shape: List[int] = None, max_params: int = int(3e5)) -> None:
        self.max_params = max_params
        if shape is None:
            self.input_size = input_size
            self.output_size = output_size
            shape = [self.input_size, max([self.input_size * 2, self.output_size * 2, 400]), self.output_size]
            # If the network is too big, shrink it
            if reduce(lambda x, y: x * y, shape) > max_params:
                hidden_size = math.floor(max_params / (self.input_size * self.output_size))

                if hidden_size > self.output_size:
                    shape = [self.input_size, hidden_size, self.output_size]
                else:
                    shape = [self.input_size, self.output_size]

        layers = []
        for ind in range(len(shape) - 1):
            layers.append(torch.nn.Linear(shape[ind], shape[ind + 1]))
            if ind < len(shape) - 2:
                layers.append(torch.nn.SELU())

        self.net = torch.nn.Sequential(*layers)
        self.to(**get_devices())

    def to(self, device: torch.device, available_devices: int) -> None:
        self.net = self.net.to(device)
        if available_devices > 1:
            self.net = torch.nn.DataParallel(self.net)

        self.device = device
        self.available_devices = available_devices

    def forward(self, input):
        with LightwoodAutocast():
            output = self.net(input)

        return output
