import torch
from torch import nn
from lightwood.mixer.helpers.default_net import DefaultNet
from lightwood.helpers.torch import LightwoodAutocast

'''
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
'''

class QClassicNet(DefaultNet):
    """
    DefaultNet variant that uses qiskit to add a final quantum layer
    """

    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 shape: list = None,
                 max_params: int = 3e7,
                 num_hidden: int = 1,
                 dropout: float = 0) -> None:
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         shape=shape,
                         max_params=max_params,
                         num_hidden=num_hidden,
                         dropout=dropout
                         )

    def to(self, device=None, available_devices=None):
        if self.ar_net:
            self.ar_net.to(device)
        return super().to(device)

    def forward(self, input):
        with LightwoodAutocast():
            if len(input.shape) == 1:
                input = input.unsqueeze(0)
            classical_output = self.net(input)
        return classical_output
