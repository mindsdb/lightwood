import torch

from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.shapes import *
from lightwood.mixers.helpers.plinear import PLinear
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.logger import log


class DefaultNet(torch.nn.Module):

    def __init__(self, dynamic_parameters,
                     input_size=None,
                     output_size=None,
                     nr_outputs=None,
                     shape=None,
                     dropout=None,
                     pretrained_net=None):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_outputs = nr_outputs
        self.max_variance = None
        # How many devices we can train this network on
        self.available_devices = 1

        self.device, _ = get_devices()
        self.dynamic_parameters = dynamic_parameters

        """
        Here we define the basic building blocks of our model,
        in forward we define how we put it all together along with an input
        """
        super(DefaultNet, self).__init__()

        if shape is None and pretrained_net is None:
            shape = [self.input_size, max([self.input_size*2,self.output_size*2,400]), self.output_size]

        if pretrained_net is None:
            log.info(f'Building network of shape: {shape}')
            rectifier = torch.nn.SELU  #alternative: torch.nn.ReLU

            layers = []
            for ind in range(len(shape) - 1):
                if (dropout is not None) and (0 < ind < len(shape)):
                    layers.append(torch.nn.Dropout(p=dropout))
                linear_function = PLinear if CONFIG.USE_PROBABILISTIC_LINEAR else torch.nn.Linear
                layers.append(linear_function(shape[ind], shape[ind+1]))
                if ind < len(shape) - 2:
                    layers.append(rectifier())

            self.net = torch.nn.Sequential(*layers)
        else:
            self.net = pretrained_net
            for layer in self.net:
                if isinstance(layer, torch.nn.Linear):
                    if self.input_size is None:
                        self.input_size = layer.in_features
                    self.output_size = layer.out_features

        self.net = self.net.to(self.device)

        if 'cuda' in str(self.device):
            self.available_devices = torch.cuda.device_count()
        else:
            self.available_devices = 1

        if self.available_devices > 1:
            self._foward_net = torch.nn.DataParallel(self.net)
        else:
            self._foward_net = self.net

    def to(self, device=None, available_devices=None):
        if device is None or available_devices is None:
            device, available_devices = get_devices()

        self.net = self.net.to(device)

        available_devices = 1
        if 'cuda' in str(device):
            available_devices = torch.cuda.device_count()

        if available_devices > 1:
            self._foward_net = torch.nn.DataParallel(self.net)
        else:
            self._foward_net = self.net

        self.device = device
        self.available_devices = available_devices

        return self

    def calculate_overall_certainty(self):
        """
        Calculate overall certainty of the model
        :return: -1 means its unknown as it is using non probabilistic layers
        """
        mean_variance = 0
        count = 0

        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                continue
            elif isinstance(layer, PLinear):

                count += 1
                mean_variance += torch.mean(layer.sigma).tolist()

        if count == 0:
            return -1  # Unknown

        mean_variance = mean_variance / count
        self.max_variance = mean_variance if self.max_variance is None \
            else mean_variance if self.max_variance < mean_variance else self.max_variance

        return (self.max_variance - mean_variance) / self.max_variance

    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return: output of the network
        """
        with LightwoodAutocast():
            output = self._foward_net(input)

        return output
