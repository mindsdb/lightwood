import math
import torch
import numpy as np
from typing import List
from sklearn.metrics import mean_absolute_error

from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.analysis.nc.nc import BaseScorer

from lightwood.model.lightgbm import LightGBM
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class SelfAware(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SelfAware, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.base_loss = 1.0

        awareness_layers = []
        awareness_net_shape = [self.input_size,
                               min(self.input_size * 2, 100),
                               1]

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
    def __init__(self, fit_params):
        super(SelfawareNormalizer, self).__init__()
        self.input_cols = list(fit_params['dtype_dict'].keys())
        self.target = fit_params['target']

        self.base_predictor = fit_params['predictor']
        self.encoders = fit_params['encoders']

        self.model = LightGBM(stop_after=30,
                              target=fit_params['target'],
                              dtype_dict=fit_params['dtype_dict'],
                              input_cols=self.input_cols,
                              fit_on_dev=False,
                              use_optuna=False)

        self.error_fn = mean_absolute_error
        self.prediction_cache = None

    def fit(self, data: List[EncodedDs], target: str) -> None:
        concat = ConcatedEncodedDs(data)
        preds = self.base_predictor(concat)
        truths = data[0].data_frame[target]
        labels = self.error_fn(preds-truths)
        data.data_frame[target] = labels

        self.model.fit(data)
        # self.model = SelfAware(data.shape[0], y.shape)
        # fit self.normalizer

    def score(self, true_input, y=None):  # @TODO: rm y
        # @TODO: make sure there is no label col here
        sa_score = self.prediction_cache if self.prediction_cache else self.model(true_input)

        if sa_score is None:
            sa_score = np.ones(true_input.shape[0])  # by default, normalizing factor is 1 for all predictions
        else:
            sa_score = np.array(sa_score)

        return sa_score
