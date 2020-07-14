from __future__ import unicode_literals, print_function, division

import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from lightwood.encoders.time_series.helpers.rnn_helpers import *
from lightwood.helpers.device import get_devices


class RnnEncoder:
    def __init__(self, encoded_vector_size=4, train_iters=75000, stop_on_error=0.8, learning_rate=0.01,
                 is_target=False, ts_n_dims=1, dropout=0.2):
        self.device, _ = get_devices()

        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._train_iters = train_iters
        self._pytorch_wrapper = torch.FloatTensor
        self._encoder = EncoderRNNNumerical(input_size=ts_n_dims, hidden_size=self._encoded_vector_size,
                                            dropout=dropout).to(self.device)
        self._decoder = DecoderRNNNumerical(output_size=ts_n_dims, hidden_size=self._encoded_vector_size,
                                            dropout=dropout).to(self.device)
        self.parameters = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=self._learning_rate)
        self.criterion = nn.MSELoss()
        self._prepared = False
        self._n_dims = ts_n_dims  # expected dimensionality of time series
        self._max_length = 20  # static decoding length
        self._sos = 0.0  # start of sequence for decoding
        self._eos = 0.0  # end of input sequence -- padding value for batches

    def to(self, device, available_devices):
        self.device = device
        self._encoder = self._encoder.to(self.device)
        return self

    def prepare_encoder(self, priming_data, feedback_hoop_function=None, batch_size=1):
        """
        The usual, run this on the initial training data for the encoder
        :param priming_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...]
        :param feedback_hoop_function: [if you want to get feedback on the training process]
        :param batch_size
        :return:
        """
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._encoder.train()
        for i in range(self._train_iters):
            average_loss = 0
            data_idx = 0

            while data_idx < len(priming_data):

                # batch building
                data_points = priming_data[data_idx:min(data_idx + batch_size, len(priming_data))]
                batch = []
                for dp in data_points:
                    data_tensor = tensor_from_series(dp, self.device, n_dims=self._n_dims, pad_value=self._eos)
                    batch.append(data_tensor)

                # shape: (batch_size, timesteps, n_dims)
                batch = torch.cat(batch, dim=0).to(self.device)
                data_idx += batch_size

                # setup loss and optimizer
                steps = batch.shape[1]
                loss = 0
                self.optimizer.zero_grad()

                # encode
                encoder_hidden = self._encoder.initHidden(self.device)
                next_tensor = data_tensor[:, 0, :].unsqueeze(dim=1)  # initial input

                for tensor_i in range(steps - 1):
                    rand = np.random.randint(2)
                    # teach from forward as well as from known tensor alternatively
                    if rand == 1:
                        next_tensor, encoder_hidden = self._encoder.forward(
                            data_tensor[:, tensor_i, :].unsqueeze(dim=1),
                            encoder_hidden)
                    else:
                        next_tensor, encoder_hidden = self._encoder.forward(next_tensor.detach(), encoder_hidden)

                    loss += self.criterion(next_tensor, data_tensor[:, tensor_i + 1, :].unsqueeze(dim=1))

                # decode
                decoder_hidden = encoder_hidden
                sos_vector = torch.full((batch.shape[0], 1, batch.shape[2]), self._sos,
                                        dtype=torch.float32).to(self.device)
                next_tensor = torch.clone(sos_vector)
                tensor_target = torch.cat([sos_vector, batch], dim=1)

                for tensor_i in range(steps - 1):
                    rand = np.random.randint(2)
                    # teach from forward as well as from known tensor alternatively
                    if rand == 1:
                        next_tensor, decoder_hidden = self._decoder.forward(
                            tensor_target[:, tensor_i, :].unsqueeze(dim=1),
                            decoder_hidden)
                    else:
                        next_tensor, decoder_hidden = self._decoder.forward(next_tensor.detach(), decoder_hidden)

                    loss += self.criterion(next_tensor, tensor_target[:, tensor_i + 1, :].unsqueeze(dim=1))

                average_loss += int(loss)
                loss.backward()
                self.optimizer.step()

            average_loss = average_loss / len(priming_data)

            if average_loss < self._stop_on_error:
                break
            if feedback_hoop_function is not None:
                feedback_hoop_function("epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(
                    epoch_n=i + 1,
                    total=self._train_iters,
                    average_loss=round(average_loss, 2)))

        self._prepared = True

    def _encode_one(self, data, initial_hidden=None, return_next_value=False):
        """
        This method encodes one single row of serial data
        :param data: multidimensional time series as list of lists [[dim1_data], [dim2_data], ...]
                     (dim_data: string with format "x11, x12, ... x1n")
        :param initial_hidden: if you want to encode from an initial hidden state other than 0s
        :param return_next_value:  if you want to return the next value in the time series too

        :return:  either encoded_value or (encoded_value, next_value)
        """
        self._encoder.eval()
        with torch.no_grad():
            data_tensor = tensor_from_series(data, self.device, n_dims=self._n_dims, pad_value=self._eos)
            steps = data_tensor.shape[1]
            encoder_hidden = self._encoder.initHidden(self.device)
            encoder_hidden = encoder_hidden if initial_hidden is None else initial_hidden

            next_tensor = None
            for tensor_i in range(steps):
                next_tensor, encoder_hidden = self._encoder.forward(data_tensor[:, tensor_i, :].unsqueeze(dim=0),
                                                                    encoder_hidden)

        if return_next_value:
            return encoder_hidden, next_tensor
        else:
            return encoder_hidden

    def encode(self, column_data, get_next_count=None):
        """
        Encode a list of time series data
        :param column_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...] to encode
        :param get_next_count: default None, but you can pass a number X and it will return the X following predictions
                               on the series for each ts_data_point in column_data
        :return: a list of encoded time series or if get_next_count !=0 two lists (encoded_values, projected_numbers)
        """

        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        for val in column_data:
            ret = []
            next = []

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
                    next_i += [next_reading.cpu()]

                next += [torch.stack(next_i)]

            ret += [encoded[0][0].cpu()]

        if get_next_count is None:
            return self._pytorch_wrapper(torch.stack(ret))
        else:
            return self._pytorch_wrapper(torch.stack(ret)), self._pytorch_wrapper(torch.stack(next))

    def _decode_one(self, hidden):
        """
        Decodes a single time series from its encoded representation.
        :param hidden: time series embedded representation tensor, with size self._encoded_vector_size
        :return: decoded time series list
        """
        self._decoder.eval()
        with torch.no_grad():
            ret = []
            next_tensor = torch.full((1, 1, self._n_dims), self._sos, dtype=torch.float32).to(self.device)
            for _ in range(self._max_length):
                next_tensor, hidden = self._decoder.forward(next_tensor, hidden)
                ret.append(next_tensor)
            return torch.stack(ret)

    def decode(self, encoded_data):
        """
        Decode a list of embedded multidimensional time series
        :param encoded_data: a list of embeddings [ e1, e2, ...] to be decoded into time series
        :return: a list of reconstructed time series
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        ret = []
        for _, val in enumerate(encoded_data):
            hidden = torch.unsqueeze(torch.unsqueeze(val, dim=0), dim=0).to(self.device)
            reconstruction = self._decode_one(hidden).cpu().squeeze().T.tolist()
            ret += [reconstruction]

        return self._pytorch_wrapper(ret)


# only run the test if this file is called from debugger
if __name__ == "__main__":
    from lightwood.encoders.time_series.helpers.test_ts import train_and_eval

    # set log, fix random seed for reproducibility
    logging.basicConfig(level=logging.DEBUG)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in str(get_devices()[0]):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # test: overfit single multi dimensional time series
    logging.info(" [Test] Testing multi-dimensional time series encoding")
    series = [[['1 2 3 4 5 6'], ['2 3 4 5 6 7'], ['3 4 5 6 7 8'], ['4 5 6 7 8 9']]]

    data = 100 * series
    n_dims = max([len(q) for q in data])

    queries = [[['1 2 3'], ['2 3 4'], ['3 4 5'], ['4 5 6']]]
    answers = [[4, 5, 6, 7]]

    params = {'answers': answers,
              'hidden_size': 10,
              'batch_size': 1,
              'dropout': 0.1,
              'ts_n_dims': n_dims,
              'train_iters': 15,
              'margin': 0.25,  # error tolerance
              'feedback_fn': lambda x: logging.info(x),
              'pred_qty': 1}

    train_and_eval(data, queries, params)
