from __future__ import unicode_literals, print_function, division

from lightwood.encoders.time_series.helpers.rnn_helpers import *
from lightwood.config.config import CONFIG
from lightwood.helpers.device import get_devices

import torch
import torch.nn as nn
from torch import optim
import numpy as np


class RnnEncoder:

    def __init__(self, encoded_vector_size=4, train_iters=75000, stop_on_error=0.8, learning_rate=0.01, is_target=False):
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._train_iters = train_iters
        self._pytorch_wrapper = torch.FloatTensor
        self._encoder = None
        self._prepared = False

        self.device, _ = get_devices()

    def to(self, device, available_devices):
        self.device = device
        self._encoder = self._encoder.to(self.device)
        return self
    
    def prepare_encoder(self, priming_data, feedback_hoop_function = None):
        """
        The usual, run this on the initial training data for the encoder
        :param priming_data: a list of lists [[time_series], ...]
        :param feedback_hoop_function: [if you wan to get feedback on the training process]
        :return:
        """
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._encoder = EncoderRNNNumerical(hidden_size=self._encoded_vector_size).to(self.device)
        optimizer = optim.Adam(self._encoder.parameters(), lr=self._learning_rate)
        criterion = nn.MSELoss()

        self._encoder.train()
        for i in range(self._train_iters):
            average_loss = 0
            for data_point in priming_data:
                data_tensor = tensor_from_series(data_point, self.device)
                loss = 0

                optimizer.zero_grad()
                encoder_hidden = self._encoder.initHidden(self.device)
                next_tensor = data_tensor[0]
                for tensor_i in range(len(data_tensor)-1):
                    rand = np.random.randint(2)
                    # teach from forward as well as from known tensor alteratively
                    if rand == 1:
                        next_tensor, encoder_hidden = self._encoder.forward(data_tensor[tensor_i] , encoder_hidden)
                    else:
                        next_tensor, encoder_hidden = self._encoder.forward(next_tensor.detach(), encoder_hidden)
                    loss += criterion(next_tensor, data_tensor[tensor_i+1])


                loss = loss
                average_loss += int(loss)
                loss.backward()
                optimizer.step()

            average_loss = average_loss/len(priming_data)

            if average_loss < self._stop_on_error:
                break
            if feedback_hoop_function is not None:
                feedback_hoop_function("epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(average_loss=average_loss, epoch_n=i+1, total=self._train_iters))

        self._prepared = True

    def _encode_one(self, data, initial_hidden = None, return_next_value = False):
        """
        This method encodes one single row of serial data
        :param data: a string representing a list of values separate by space, for example: `1 2 3 4` or a list [1, 2, 3, 4]
        :param initial_hidden: if you want to encode from an initial hidden state other than 0s
        :param return_next_value:  if you to return the next value in the time series too

        :return:  either encoded_value or encoded_value, next_value
        """
        self._encoder.eval()
        with torch.no_grad():

            data_tensor = tensor_from_series(data, self.device)

            encoder_hidden = self._encoder.initHidden(self.device)
            encoder_hidden = encoder_hidden if initial_hidden is None else initial_hidden
            if type(encoder_hidden) is list:
                encoder_hidden = torch.Tensor([[encoder_hidden]], device=device)
            next_tensor = None
            for tensor_i in range(len(data_tensor)):
                next_tensor, encoder_hidden = self._encoder.forward(data_tensor[tensor_i], encoder_hidden)

        if return_next_value:
            return encoder_hidden, next_tensor
        else:
            return encoder_hidden


    def encode(self, column_data, get_next_count = None):
        """
        Encode a list of time series data
        :param column_data: a list of lists [ [ts_data_point_list1], ...] to encode
        :param get_next_count: default None, but you can pass a number X and it will return the X following predictions on the series for each ts_data_point in column_data
        :return: a list of encoded values or if get_next_count !=0 two lists encoded_values, projected_numbers
        """

        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')


        ret = []
        next = []

        for val in column_data:
            if get_next_count is None:
                encoded = self._encode_one(val)
            else:
                if get_next_count <= 0:
                    raise Exception('get_next_count must be greater than 0')

                hidden = None
                vector = val

                next_i = []

                for j in range(get_next_count):

                    hidden, next_reading = self._encode_one(vector, initial_hidden=hidden, return_next_value=True)
                    vector = [next_reading]
                    if j == 0:
                        encoded = hidden
                    next_i += [next_reading]

                next += [next_i[0][0].cpu()]

            ret += [encoded[0][0].cpu()]

        if get_next_count is None:
            return self._pytorch_wrapper(torch.stack(ret))
        else:
            return self._pytorch_wrapper(torch.stack(ret)), self._pytorch_wrapper(torch.stack(next))



# only run the test if this file is called from debugger
if __name__ == "__main__":
    series = []
    for j in range(100):
        length = 3 + np.random.randint(14)
        skip = 1+np.random.randint(4)

        start = np.random.randint(30)
        vec = [start+j*skip for j in range(length)]

        series+=[' '.join([str(x) for x in vec])]

    encoder = RnnEncoder(encoded_vector_size=3,train_iters=10)
    encoder.prepare_encoder(series, feedback_hoop_function=lambda x:print(x))


    # test de decoder
    init_vector = ['31 33 35 37', '1 2 3 4 5 6']

    print(encoder.encode(column_data=init_vector, get_next_count=2))
    print(encoder.encode(column_data=init_vector))
