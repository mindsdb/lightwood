import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from lightwood.config.config import CONFIG
from lightwood.helpers.device import get_devices


class PLinear(nn.Module):
    '''
    Implementation of probabilistic weights via Linear function
    '''

    def __init__(self, in_features, out_features, bias=True):
        """

        :param in_features:  as name suggests
        :param out_features: this essentially the number of neurons
        :param bias: if you want a specific bias
        """

        super(PLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # these are the matrices that we will optimize for
        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        self.mean = Parameter(torch.Tensor(out_features, in_features))

        # there can be various ways to sample, given various distributions,
        # we will stick with discrete normal as it is way faster
        self.w_sampler = self.w_discrete_normal

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # make sure that we tell the graph that these two need to be optimized
        self.sigma.requiresGrad = True  # set requiresGrad to true!
        self.mean.requiresGrad = True  # set requiresGrad to true!

        self.device, _ = get_devices()

    def reset_parameters(self):
        """
        This sets the initial values for the distribution parameters, mean, sigma

        """

        init.kaiming_uniform_(self.mean, a=math.sqrt(5))  # initial means (just as in original linear)
        init.uniform_(self.sigma, a=0.05, b=0.5)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def w_discrete_normal(self):
        """
        Sample a w matrix based on a discrete normal distribution
        :return: w
        """

        # multipliers
        sigma_multiplier = np.random.choice([1, 2, 3, 4], p=[0.41, 0.46, 0.09, 0.04])
        sigma_multiplier = torch.Tensor([sigma_multiplier])
        sigma_multiplier = sigma_multiplier.to(self.device)

        # generate the initial tensor, this will ultimately transforms in to the weights
        w = torch.Tensor(self.out_features, self.in_features)

        w = w.to(self.device)

        # make sure that they are evently distributed between -1, 1
        init.uniform_(w, a=-1, b=1)

        # torch.div(sigma_multiplier,torch.var(2).to(self.device))

        # adjust based on sigma

        w = torch.mul(
            self.mean.to(self.device),
            torch.add(
                1,
                torch.mul(
                    torch.mul(w, torch.abs(self.sigma).to(self.device)),
                    torch.div(sigma_multiplier, 2)
                )
            )
        )

        # you can see how the average sigma changes over trainings
        # print(torch.mean(self.sigm.to(self.device)a))
        return w

    def forward(self, input):
        """
        Forward pass of the function.
        The goal is to generate values such as if they weights of the linear operation are sampled
        from a normal distribution
        """

        return F.linear(input, self.w_sampler(), self.bias)


if __name__ == "__main__":

    pass
