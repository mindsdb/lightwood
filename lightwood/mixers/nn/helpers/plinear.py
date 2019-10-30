# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo
from torch.nn import init
import math
import random
import numpy as np

class PLinear(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, out_features, bias=True):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(PLinear ,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        self.mean = Parameter(torch.Tensor(out_features, in_features))
        self.w = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)


        self.sigma.requiresGrad = True # set requiresGrad to true!
        self.mean.requiresGrad = True  # set requiresGrad to true!

    def reset_parameters(self):
        init.kaiming_uniform_(self.mean, a=math.sqrt(5)) # initial means (just as in original linear)
        init.uniform_(self.sigma, a=0.4, b=2) # initial sigmas from 0.5-2

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        '''
        Forward pass of the function.
        The goal is to generate values such as if they weights of the linear operation are sampled from a normal distribution
        '''

        # for this we generate matrices for means and sigmas in distributions on the Probablility axis of the probability distribution for the weights

        mu_y = 1 / ((self.sigma ** 2) * math.pi * 2) ** (0.5)
        sigma_y = torch.sqrt(mu_y - mu_y * (math.e ** (-0.5)))
        mu_y = mu_y.tolist()
        sigma_y = sigma_y.tolist()

        # now we can generate samples 

        y_rand = torch.Tensor([[float(abs(np.random.normal(mu_y[y_i][x_i], sigma_y[y_i][x_i], 1)[0] - mu_y[y_i][x_i]))
                                for x_i, x in enumerate(y)] for y_i, y in enumerate(mean)])

        # here we will generate a matrix of negative and positive random numbers as we need to generate numbers on either side of mean
        x_sign = torch.Tensor(
            [[1 if random.randint(0, 1) > 0 else -1 for x_i, x in enumerate(y)] for y_i, y in enumerate(self.mean)])

        # now we bind the sigmas and means to the graph so they can be aligned in backpropagation
        self.w = mean + torch.sqrt(torch.abs(-2 * (self.sigma ** 2) * torch.log(y_rand * torch.sqrt(2 * np.pi * (self.sigma ** 2))))) * x_sign

        return F.linear(input, self.w, self.bias)


if __name__ == "__main__":
    import time

    mean = torch.Tensor(10, 5)
    sigma = torch.Tensor(10, 5)



    x_sign = torch.Tensor([[1 if random.randint(0,1) > 0 else -1 for x_i, x in enumerate(y)] for  y_i, y in enumerate(mean)])


    #print(mean)
    init.uniform_(mean, a=-1, b=10)#kaiming_uniform_(mean, a=math.sqrt(5))
    init.uniform_(sigma, a=0.4, b=2)

    #print(mean)
    #print(sigma_sq)

    print(mean)
    print(sigma)
    mu_y = 1 / ((sigma**2) * math.pi * 2) ** (0.5)
    print("mu_y")
    print(mu_y)
    sigma_y = torch.sqrt(mu_y - mu_y * (math.e ** (-0.5)))
    mu_y = mu_y.tolist()
    print(sigma_y)
    sigma_y = sigma_y.tolist()



    y_rand = torch.Tensor([[float(abs(np.random.normal(mu_y[y_i][x_i], sigma_y[y_i][x_i], 1)[0] - mu_y[y_i][x_i])) for x_i, x in enumerate(y)] for y_i, y in enumerate(mean)])

    x_rand = mean + torch.sqrt(torch.abs(-2 * (sigma**2) * torch.log(y_rand * torch.sqrt(2 * np.pi * (sigma**2))))) * x_sign

    print(x_rand)


