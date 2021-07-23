import math
import torch
import numpy as np
from typing import Union

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from lightwood.analysis.nc.nc import BaseScorer
from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.data.encoded_ds import ConcatedEncodedDs


class SelfawareNormalizer(BaseScorer):
    def __init__(self, fit_params):
        super(SelfawareNormalizer, self).__init__()

        self.input_cols = list(fit_params['dtype_dict'].keys())
        self.base_predictor = fit_params['predictor']
        self.encoders = fit_params['encoders']
        self.target = fit_params['target']

        self.model = ElasticNet()
        self.prediction_cache = None
        self.error_fn = mean_absolute_error

    def fit(self, data: ConcatedEncodedDs, target: str) -> None:
        if data and target:
            preds = self.base_predictor(data)
            truths = data.data_frame[target]
            labels = abs(preds.values.squeeze() - truths.values)
            data.data_frame[target] = labels
            enc_data = data.get_encoded_data(include_target=False).numpy()
            self.model.fit(enc_data, labels)

    def predict(self, data: Union[ConcatedEncodedDs, torch.Tensor]) -> np.ndarray:
        if isinstance(data, ConcatedEncodedDs):
            data = data.get_encoded_data(include_target=False)
        return self.model.predict(data.numpy())

    def score(self, true_input, y=None):
        sa_score = self.prediction_cache if self.prediction_cache is not None else self.model.predict(true_input)

        if sa_score is None:
            sa_score = np.ones(true_input.shape[0])  # by default, normalizing factor is 1 for all predictions
        else:
            sa_score = np.array(sa_score)  # @TODO: try 0.5+softmax(x)

        return sa_score


class SelfAwareNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        """Unused alternative to ElasticNet regression model, aims to predict
        the error that the main predictor will yield on each prediction"""
        super(SelfAwareNet, self).__init__()

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