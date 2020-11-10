import os
import sys
import unittest

import numpy as np
import torch

from lightwood.helpers.device import get_devices
from lightwood.encoders.time_series.cnn import *
from lightwood.encoders.time_series.helpers.cnn_helpers import (
            tensor_from_series, 
            simple_data_generator,
            nonlin_data_generator, 
            random_data_generator)
 
class TestCnnEncoder(unittest.TestCase):
    def __init__(self, blocks=[1], kernel_size=3, train_epochs=100, learning_rate=0.001):
        super(TestCnnEncoder, self).__init__()
        self._kernel_size = kernel_size
        self._blocks = blocks
        self._train_epochs = train_epochs
        self._learning_rate = learning_rate

    def initial(self):
        n_dims = 2
        length = 400
        data = simple_data_generator(length, n_dims)
        data = 200*data
        batch_size = 32

        self._train_epochs = 40
        self._learning_rate = 0.005

        self._blocks = [32,16,8,1]
        self._kernel_size = 5

        encoder = CnnEncoder(blocks=self._blocks, 
                             kernel_size = self._kernel_size, 
                             train_epochs = self._train_epochs, 
                             batch_size = batch_size, 
                             ts_n_dims=n_dims,
                             learning_rate=self._learning_rate)
        final_loss = encoder.prepare_encoder(data, 
                feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
        # encoded = encoder.encode(data[0])  # encode one sample
        # decoded = encoder.decode(encoded)
        return final_loss#, encoded, decoded

    def nonlinear(self):
        n_dims = 1
        length = 800
        data = nonlin_data_generator(length, n_dims)
        data = 500*data
        batch_size = 32

        self._train_epochs = 50
        self._learning_rate = 0.005

        self._blocks = [128,64,32,8,2]
        self._kernel_size = 5

        encoder = CnnEncoder(blocks=self._blocks,
                             kernel_size = self._kernel_size, 
                             train_epochs = self._train_epochs, 
                             batch_size = batch_size, 
                             ts_n_dims=n_dims, 
                             learning_rate=self._learning_rate)
        final_loss = encoder.prepare_encoder(data, 
                feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
        return final_loss

    def random(self):
        n_dims = 1
        length = 400
        data = random_data_generator(length, n_dims)
        data = 500*data
        batch_size = 32

        self._train_epochs = 30
        self._learning_rate = 0.005

        self._blocks = [32,16,8,1]
        self._kernel_size = 5

        encoder = CnnEncoder(blocks=self._blocks,
                             kernel_size = self._kernel_size,
                             train_epochs = self._train_epochs, 
                             batch_size = batch_size, 
                             ts_n_dims=n_dims, 
                             learning_rate=self._learning_rate)
        final_loss = encoder.prepare_encoder(data, 
                feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
        return final_loss


### Initial test, time series is of form [[0,1,2,3,...n],[20,21,22,..n+20]]
# Very easy task, linear relationship, performs well, especially with many samples
test = TestCnnEncoder()
result = test.initial() 

### 2nd test, random sequence of numbers in range (0,100), each sample is the identical
# Aimed to test long term memory of model
# Converges, but loss is still high
# test = TestCnnEncoder()
# result = test.random()

### 3rd test, non-linear function y = x^3 + x^2 + c
# Where, x is value in dimension, c is dimension
# Performs very poorly, more complex model required
# test = TestCnnEncoder()
# result = test.nonlinear()






