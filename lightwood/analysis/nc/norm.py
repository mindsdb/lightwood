import math
from typing import Union

import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.linear_model import ElasticNet, SGDRegressor, Ridge  # @TODO: settle on one of these
from sklearn.metrics import mean_absolute_error

from lightwood.api.dtype import dtype
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
        self.target_dtype = fit_params['dtype_dict'][fit_params['target']]
        self.multi_ts_task = fit_params['is_multi_ts']

        self.model = Ridge()  # SGDRegressor()  # ElasticNet()
        self.prediction_cache = None
        self.error_fn = mean_absolute_error

    def fit(self, data: ConcatedEncodedDs, target: str) -> None:
        if data and target:
            preds = self.base_predictor(data, predict_proba=True)
            truths = data.data_frame[target]
            labels = self.get_labels(preds, truths.values, data.encoders[self.target])
            enc_data = data.get_encoded_data(include_target=False).numpy()
            self.model.fit(enc_data, labels)

    def predict(self, data: Union[ConcatedEncodedDs, torch.Tensor]) -> np.ndarray:
        if isinstance(data, ConcatedEncodedDs):
            data = data.get_encoded_data(include_target=False)
        raw = self.model.predict(data.numpy())
        clipped = np.clip(raw, 0.1, 1e4)  # set limit deviations (@TODO: benchmark stability)
        # smoothed = clipped / clipped.mean()
        return clipped

    def score(self, true_input, y=None):
        sa_score = self.prediction_cache if self.prediction_cache is not None else self.model.predict(true_input)

        if sa_score is None:
            sa_score = np.ones(true_input.shape[0])  # by default, normalizing factor is 1 for all predictions
        else:
            sa_score = np.array(sa_score)

        return sa_score

    def get_labels(self, preds: pd.DataFrame, truths: np.ndarray, target_enc):
        if self.target_dtype in [dtype.integer, dtype.float]:

            if not self.multi_ts_task:
                preds = preds.values.squeeze()
            else:
                preds = [p[0] for p in preds.values.squeeze()]

            # abs(np.min(np.log(labels))) + (np.log(labels) / np.mean(np.log(labels)))
            # abs(np.min(diffs / np.mean(diffs))) + (diffs / np.mean(diffs))
            # diffs = np.log(abs(preds - truths))
            diffs = np.square(abs(preds - truths))
            labels = diffs
            # diffs = diffs / np.mean(diffs)
            # labels = diffs + abs(np.min(diffs))

        elif self.target_dtype in [dtype.binary, dtype.categorical]:
            prob_cols = [col for col in preds.columns if '__mdb_proba' in col]
            col_names = [col.replace('__mdb_proba_', '') for col in prob_cols]
            if prob_cols:
                preds = preds[prob_cols]
                if '__mdb_proba___mdb_unknown_cat' in preds.columns:
                    preds.pop('__mdb_proba___mdb_unknown_cat')
                    prob_cols.remove('__mdb_proba___mdb_unknown_cat')
                    col_names.remove('__mdb_unknown_cat')

            # reorder preds to ensure classes are in same order as in target_enc
            preds.columns = col_names
            new_order = [v for k, v in sorted(target_enc.rev_map.items(), key=lambda x: x[0])]
            preds = preds.reindex(columns=new_order)

            # get log loss
            preds = preds.values.squeeze()
            preds = preds if prob_cols else target_enc.encode(preds).tolist()
            truths = target_enc.encode(truths).numpy()
            labels = entropy(truths, preds, axis=1)

        else:
            raise(Exception(f"dtype {self.target_dtype} not supported for confidence normalizer"))

        return labels


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