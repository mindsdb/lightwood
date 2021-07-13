import math
import torch

from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.analysis.nc.nc import BaseScorer


class SelfAware(torch.nn.Module):
    def __init__(self, input_size, output_size, nr_outputs):
        super(SelfAware, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.nr_outputs = nr_outputs
        self.base_loss = 1.0

        awareness_layers = []
        awareness_net_shape = [self.input_size,
                               min(self.input_size * 2, 100),
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

    def forward(self, true_input):
        """
        :param true_input: tensor with data point features
        :param main_net_output: tensor with main NN prediction for true_input
        :return: predicted loss value over the tensor samples
        """
        with LightwoodAutocast():
            aware_in = true_input
            output = self.net(aware_in)
            return output


class SelfawareNormalizer(BaseScorer):
    def __init__(self, fit_params=None):
        super(SelfawareNormalizer, self).__init__()
        self.prediction_cache = None
        self.output_column = fit_params['output_column']

    def fit(self, x, y):
        """ No fitting is needed, as the self-aware model is trained in Lightwood """
        pass

    def score(self, true_input, y=None):
        sa_score = self.prediction_cache

        if sa_score is None:
            sa_score = np.ones(true_input.shape[0])  # by default, normalizing factor is 1 for all predictions
        else:
            sa_score = np.array(sa_score)

        return sa_score
