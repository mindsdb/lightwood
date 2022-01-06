import math
from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from lightwood.helpers.log import log
from lightwood.helpers.device import get_devices
from lightwood.encoder.categorical.gym import Gym
from lightwood.analysis.base import BaseAnalysisBlock


class PLinearWrapper(BaseAnalysisBlock):
    """
    Wraps a probablistic NN that outputs distribution parameters over the residual between
    true and predicted values. Confidence score and bounds can be extracted based on repeated
    sampling / bootstraping.

    Given (X, yhat) -> learn param distributions to model residuals |y - yhat|.

    Two inference modes are provided:
        - quantile: sort sampled predictions and pick bounds based on the i-th sample given confidence level `i`
        - normal: derive symmetric bounds assuming a normal distribution and point predictions as mean response
    """
    def __init__(self, mode='quantile'):
        super().__init__()
        self.mode = mode
        self.model = None
        self.trainer = None
        self.optimizer = None
        self.time_budget = 30
        self.n_samples = 100  # 100 is the minimum for arbitrary quantiles at 0.01 resolution
        self.device, _ = get_devices()
        self.pct_train = 0.8
        self.pct_test = 1 - self.pct_train
        self.callback = lambda err, y, z: log.debug(f'PLinear error is {err}')
        assert self.pct_train + self.pct_test == 1.0

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        X = ns.encoded_val_data.get_encoded_data(include_target=False)
        y = torch.tensor(ns.encoded_val_data.get_column_original_data(ns.target).values).unsqueeze(1)
        yh = torch.tensor(ns.normal_predictions['prediction'].values).unsqueeze(1)
        r = (y - yh).float()
        X = torch.hstack([X, yh]).float()  # predicted value is the most important prior

        mask = torch.rand((len(y),)) < 0.8
        train_X = X[mask, :]
        test_X = X[~mask, :]
        train_y = r[mask]
        test_y = r[~mask]

        n_targets = 1
        n_features = X.shape[1]
        self.model = PLinear(in_features=n_features, out_features=n_targets)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        loss_fn = torch.nn.MSELoss()
        scheduler = None

        self.trainer = Gym(self.model, self.optimizer, scheduler, loss_fn, self.device)
        train_data_loader = DataLoader(list(zip(train_X, train_y)),
                                       batch_size=min(32, len(train_X)),
                                       shuffle=True)
        test_data_loader = DataLoader(list(zip(test_X, test_y)),
                                      batch_size=min(32, len(test_X)),
                                      shuffle=True)

        self.trainer.fit(train_data_loader, test_data_loader,
                         desired_error=0.01,
                         max_time=self.time_budget,
                         callback=self.callback)

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:

        ns = SimpleNamespace(**kwargs)
        conf_level = ns.pred_args.fixed_confidence
        if conf_level is None:
            conf_level = 0.95

        X = ns.encoded_data
        yh = torch.tensor(ns.predictions['prediction'].values).unsqueeze(1)
        X = torch.hstack([X, yh]).float()

        samples = []
        for _ in range(self.n_samples):
            samples.append(self.model.forward(X))

        samples = torch.stack(samples).squeeze()
        if self.mode == 'quantile':
            sorted_samples = torch.sort(samples, dim=0).values
            sorted_samples = sorted_samples - sorted_samples.median(dim=0).values  # rescale so that median is null
            min_bound = sorted_samples[round(samples.shape[0] * (1.0 - conf_level)), :]
            max_bound = sorted_samples[round(samples.shape[0] * conf_level), :]
            row_insights['lower'] = row_insights['prediction'] + min_bound.detach().numpy()
            row_insights['upper'] = row_insights['prediction'] + max_bound.detach().numpy()
        else:
            stds = torch.std(samples, dim=0)
            mean = torch.tensor(row_insights['prediction'])
            min_bound = mean - ((conf_level / 2 * stds) / np.sqrt(len(samples)))
            max_bound = mean + ((conf_level / 2 * stds) / np.sqrt(len(samples)))
            row_insights['lower'] = min_bound.detach().numpy()
            row_insights['upper'] = max_bound.detach().numpy()

        row_insights['confidence'] = conf_level

        return row_insights, global_insights


class PLinear(torch.nn.Module):
    """
    Implementation of probabilistic weights via Linear function
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.sigma = Parameter(torch.Tensor(out_features, in_features))

        # there can be various distributions & ways to sample, we stick with discrete normal as it is way faster
        self.w_sampler = self.w_discrete_normal

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # make sure that we tell the graph that these two need to be optimized
        self.sigma.requiresGrad = True

    def reset_parameters(self):
        torch.nn.init.uniform_(self.sigma, a=0.05, b=0.2)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.sigma)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def w_discrete_normal(self):
        """
        Sample a w matrix based on a discrete normal distribution
        :return: w
        """
        sigma_multiplier = np.random.choice([1, 2, 3, 4], p=[0.41, 0.46, 0.09, 0.04])

        device_str, _ = get_devices()

        # generate the initial tensor, this will ultimately transform into the weights
        w = torch.Tensor(self.out_features, self.in_features)

        device = torch.device(device_str)
        w.to(device)

        # make sure that they are evenly distributed between -1, 1
        torch.nn.init.uniform_(w, a=-1, b=1)

        # adjust based on sigma
        w = 1 + w * torch.abs(self.sigma) * sigma_multiplier

        return w

    def forward(self, input):
        """
        Generate values such as if the weights of the linear operation were sampled from a normal distribution.
        """
        return F.linear(input, self.w_sampler(), self.bias)
