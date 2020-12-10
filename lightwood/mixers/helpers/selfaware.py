import torch
import math

from lightwood.config.config import CONFIG
from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast


class SelfAware(torch.nn.Module):
    def __init__(self, input_size, output_size, nr_outputs):
        super(SelfAware, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.nr_outputs = nr_outputs

        awareness_layers = []
        awareness_net_shape = [(self.input_size + self.output_size),
                               max([int((self.input_size + self.output_size) * 1.5), 300]),
                               self.nr_outputs]

        for ind in range(len(awareness_net_shape) - 1):
            rectifier = torch.nn.SELU
            awareness_layers.append(torch.nn.Linear(awareness_net_shape[ind], awareness_net_shape[ind + 1]))
            if ind < len(awareness_net_shape) - 2:
                awareness_layers.append(rectifier())

        self.net = torch.nn.Sequential(*awareness_layers)

        for layer in self.net:
            if hasattr(layer, 'weight'):
                torch.nn.init.normal_(layer.weight, mean=0., std=1 / math.sqrt(layer.out_features))
            if hasattr(layer, 'bias'):
                torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

        self.device, self.available_devices = get_devices()
        self.to(self.device, self.available_devices )

    def to(self, device, available_devices):
        if available_devices > 1:
            self.net = torch.nn.DataParallel(self.net).to(device)
        else:
            self.net = self.net.to(device)

        return self

    def forward(self, true_input, main_net_output):
        """
        :param true_input: tensor with data point features
        :param main_net_output: tensor with main NN prediction for true_input
        :return: predicted loss value over the tensor samples
        """
        with LightwoodAutocast():
            aware_in = torch.cat((true_input, main_net_output), 1)
            output = self.net(aware_in)
            return output
    