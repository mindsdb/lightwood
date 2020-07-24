import torch
import math

from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.ranger import Ranger


# Todo: conform to style guidelines
class SelfAware(torch.nn.Module):
    def __init__(self, input_size, output_size, nr_outputs):
        super(SelfAware, self).__init__()

        self.available_devices = None
        self.device = None

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

        self.awareness_net = torch.nn.Sequential(*awareness_layers)

        for layer in self.awareness_net:
            if hasattr(layer, 'weight'):
                torch.nn.init.normal_(layer.weight, mean=0., std=1 / math.sqrt(layer.out_features))
            if hasattr(layer, 'bias'):
                torch.nn.init.normal_(layer.bias, mean=0., std=0.1)

        self.opt = Ranger(self.awareness_net.parameters())

    def to(self, device, available_devices):
        self.awareness_net = self.awareness_net.to(device)

        if available_devices > 1:
                self.awareness_net = torch.nn.DataParallel(self.awareness_net)

        self.device = device
        self.available_devices = available_devices

        return self

    def forward(self, input):
        """
        :param input: tensor with (true_input, main_net_output) pairs to estimate loss magnitude
        :return: predicted loss value over the tensor samples
        """
        output = self.awareness_net(input)
        return output



if __name__ == '__main__':
    print("Various tests that need to be aggregated in tests/ before PRing")
    sa = SelfAware(100, 50, 1)

    print(sa)