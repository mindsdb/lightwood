
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
        """

        :param in_features:  as name suggests
        :param out_features: this essentially the number of neurons
        :param bias: if you want a specific bias
        """

        super(PLinear ,self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # these are the matrices that we will optimize for
        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        self.mean = Parameter(torch.Tensor(out_features, in_features))

        # there can be various ways to sample, given various distributions, we will stick with discrete normal as it is way faster
        self.w_sampler = self.w_discrete_normal

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        #make sure that we tell the graph that these two need to be optimized
        self.sigma.requiresGrad = True # set requiresGrad to true!
        self.mean.requiresGrad = True  # set requiresGrad to true!


    def reset_parameters(self):
        """
        This sets the initial values for the distribution parameters, mean, sigma

        """
        if self.w_sampler == self.w_normal:
            init.kaiming_uniform_(self.mean, a=math.sqrt(5)) # initial means (just as in original linear)
            init.uniform_(self.sigma, a=0.01, b=0.1) # initial sigmas from 0.5-2
        elif self.w_sampler == self.w_discrete_normal:
            init.kaiming_uniform_(self.mean, a=math.sqrt(5))  # initial means (just as in original linear)
            init.uniform_(self.sigma, a=0.05, b=0.2)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def w_discrete_normal(self):
        """
        Sample a w matrix based on a discrete normal distribution
        :return: w
        """

        bucket_seed = random.uniform(0,1) # gett a number that can help determines which bucket of the distribution can numbers fall in
        sigma_multiplier = 1 # this tell it how wide do we stretch the random saples

        # set the widths for sampling, based on the seed
        if bucket_seed > 0.33:
            sigma_multiplier = 2
            if bucket_seed > 0.8:
                sigma_multiplier = 4

        # generate the initial tensor, this will ultimately transforms in to the weights
        w = torch.Tensor(self.out_features, self.in_features)
        # make sure that they are evently distributed between -1, 1
        init.uniform_(w, a=-1, b=1)

        # adjust based on sigma
        w = self.mean*( 1+ w * torch.abs(self.sigma) * sigma_multiplier )
        # you can see how the average sigma changes over trainings
        #print(torch.mean(self.sigma))
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

        return F.linear(input, self.w_sampler(), self.bias)


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


