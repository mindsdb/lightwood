
# Import PyTorch
import torch #
import torch.nn as nn # import modules

from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F # import torch functions
from torch.nn import init
import math
import random
import numpy as np



class PLinear(nn.Module):
    '''
    Implementation of probabilistic weights via Linear function
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

        self.w_method = self.w_uniform

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        #self.w.requiresGrad = True
        self.sigma.requiresGrad = True # set requiresGrad to true!
        self.mean.requiresGrad = True  # set requiresGrad to true!

    def reset_parameters(self):
        """
        This sets the initial values for the distribution parameters, mean, sigma

        """

        if self.w_method == self.w_normal:
            init.kaiming_uniform_(self.mean, a=math.sqrt(5)) # initial means (just as in original linear)
            init.uniform_(self.sigma, a=0.01, b=0.1) # initial sigmas from 0.5-2
        elif self.w_method == self.w_uniform:
            init.kaiming_uniform_(self.mean, a=math.sqrt(5))  # initial means (just as in original linear)
            init.uniform_(self.sigma, a=0.01, b=0.1)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def w_uniform(self):
        """
        Sample a w matrix based on a uniform distribution
        :return: w
        """


        bucket_seed = random.uniform(0,1)

        sigma_multiplier = 1

        if bucket_seed > 0.66:

            sigma_multiplier = 2
        elif bucket_seed > 0.8:
            sigma_multiplier = 4
            print(sigma_multiplier)

        seed = torch.Tensor(self.out_features, self.in_features)
        init.uniform_(seed, a=-1, b=1)

        w = self.mean*( 1+ seed * self.sigma * sigma_multiplier )

        #print(w)
        #seed | w + seed*(w*sigma) = w(1+seed*sigma)

        return w


    def w_normal(self):
        """
        This samples a w based on the distribution parameter matrices self.mean, self.sigma
        @:return w
        """

        # for this we generate matrices for means and sigmas in distributions on the Probablility axis of the probability distribution for the weights
        mu_y = 1 / (torch.abs((self.sigma ** 2) * math.pi * 2)) ** (0.5)
        sigma_y = torch.sqrt(torch.abs(mu_y - mu_y * (math.e ** (-0.5))))
        mu_y = mu_y.tolist()
        sigma_y = sigma_y.tolist()

        # now we can generate samples

        y_rand = [[float(np.random.normal(mu_y[y_i][x_i], sigma_y[y_i][x_i], 1)[0]) for x_i, x in enumerate(y)] for
                  y_i, y in enumerate(sigma_y)]
        y_rand = torch.Tensor([[y_rand[y_i][x_i] if y_rand[y_i][x_i] <= mu_y[y_i][x_i] else y_rand[y_i][x_i] -
                                                                                            mu_y[y_i][x_i] for x_i, x in
                                enumerate(y)] for y_i, y in enumerate(sigma_y)])

        # print(y_rand)

        x_sign = torch.Tensor(
            [[1 if random.randint(0, 1) > 0 else -1 for x_i, x in enumerate(y)] for y_i, y in enumerate(sigma_y)])

        # now we bind the sigmas and means to the graph so they can be aligned in backpropagation
        w = self.mean + torch.sqrt(torch.abs(
            -2 * (self.sigma ** 2) * torch.log(torch.abs(y_rand * torch.sqrt(2 * np.pi * (self.sigma ** 2)))))) * x_sign

        return w

    def forward(self, input):
        '''
        Forward pass of the function.
        The goal is to generate values such as if they weights of the linear operation are sampled from a normal distribution
        '''




        return F.linear(input, self.w_method(), self.bias)


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


