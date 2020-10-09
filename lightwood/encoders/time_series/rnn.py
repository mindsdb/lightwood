from __future__ import unicode_literals, print_function, division

from lightwood.encoders.time_series.helpers.rnn_helpers import *
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.datetime import DatetimeEncoder
from lightwood.helpers.device import get_devices

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class RnnEncoder(BaseEncoder):

    def __init__(self, encoded_vector_size=100, train_iters=100, stop_on_error=0.01, learning_rate=0.01,
                 is_target=False, ts_n_dims=1):
        super().__init__(is_target)
        self.device, _ = get_devices()
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._train_iters = train_iters  # training epochs
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self._is_setup = False
        self._max_ts_length = 0
        self._sos = 0.0  # start of sequence for decoding
        self._eos = 0.0  # end of input sequence -- padding value for batches
        self._n_dims = ts_n_dims
        self._normalizer = None
        self._target_ar_normalizers = []
        self._criterion = nn.MSELoss()

    def setup_nn(self, additional_targets=None):
        """This method must be executed after initializing, else self.secondary_type is unassigned"""
        if self.secondary_type == 'datetime':
            self._normalizer = DatetimeEncoder(sinusoidal=True)
            self._n_dims *= len(self._normalizer.fields)*2  # sinusoidal datetime components
        elif self.secondary_type == 'numeric':
            self._normalizer = MinMaxNormalizer()

        total_dims = self._n_dims
        if additional_targets:
            for t in additional_targets:
                if t['original_type'] == 'categorical':
                    t['normalizer'] = CatNormalizer()
                    t['normalizer'].prepare(t['data'])
                    total_dims += len(t['normalizer'].scaler.categories_[0])
                else:
                    t['normalizer'] = MinMaxNormalizer()
                    t['normalizer'].prepare(t['data'])
                    total_dims += 1

        self._encoder = EncoderRNNNumerical(input_size=total_dims, hidden_size=self._encoded_vector_size).to(self.device)
        self._decoder = DecoderRNNNumerical(output_size=total_dims, hidden_size=self._encoded_vector_size).to(self.device)
        self._parameters = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self._optimizer = optim.AdamW(self._parameters, lr=self._learning_rate, weight_decay=1e-4)
        self._is_setup = True

    def to(self, device, available_devices):
        if self._is_setup:
            self.device = device
            self._encoder = self._encoder.to(self.device)
            self._decoder = self._decoder.to(self.device)
        return self

    def prepare(self, priming_data, previous_target_data=None, feedback_hoop_function=None, batch_size=256):
        """
        The usual, run this on the initial training data for the encoder
        :param priming_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...]
        :param previous_target_data: tensor with encoded previous target values for autoregressive tasks
        :param feedback_hoop_function: [if you want to get feedback on the training process]
        :param batch_size
        :return:
        """
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        else:
            self.setup_nn(previous_target_data)

        # Convert to array and determine max length:
        for i in range(len(priming_data)):
            if not isinstance(priming_data[i][0], list):
                priming_data[i] = [priming_data[i]]  # add dimension for 1D timeseries
            self._max_ts_length = max(len(priming_data[i][0]), self._max_ts_length)  # TODO: this is set at Native... should we still check?

        # normalize data
        if self._normalizer:
            self._normalizer.prepare(priming_data)

        if previous_target_data is not None and len(previous_target_data) > 0:
            for target_dict in previous_target_data:
                normalizer = target_dict['normalizer']
                target_dict['encoded_data'] = normalizer.encode(target_dict['data'])
                self._target_ar_normalizers.append(normalizer)

        # decrease batch_size for small datasets
        if batch_size >= len(priming_data):
            batch_size = len(priming_data) // 2

        self._encoder.train()
        for i in range(self._train_iters):
            average_loss = 0
            data_idx = 0

            while data_idx < len(priming_data):

                # batch building, shape: (batch_size, timesteps, n_dims)
                data_points = priming_data[data_idx:min(data_idx + batch_size, len(priming_data))]
                batch = []
                for dp in data_points:
                    data_tensor = tensor_from_series(dp, self.device, self._n_dims,
                                                     self._eos, self._max_ts_length,
                                                     self._normalizer)
                    batch.append(data_tensor)
                batch = torch.cat(batch, dim=0).to(self.device)

                # include autoregressive target data
                if previous_target_data is not None and len(previous_target_data) > 0:
                    for target_dict in previous_target_data:
                        t_dp = target_dict['encoded_data'][data_idx:min(data_idx + batch_size, len(priming_data))]
                        target_tensor = torch.Tensor(t_dp).to(self.device)
                        target_tensor[torch.isnan(target_tensor)] = 0.0
                        if len(t_dp.shape) < 3:
                            target_tensor = target_tensor.unsqueeze(2)

                        # concatenate descriptors
                        batch = torch.cat((batch, target_tensor), dim=-1)

                # setup loss and optimizer
                self._optimizer.zero_grad()
                data_idx += batch_size
                steps = batch.shape[1]
                loss = 0

                # encode
                encoder_hidden = self._encoder.initHidden(self.device, batch_size=batch.shape[0])
                next_tensor = batch[:, 0, :].unsqueeze(dim=1)  # initial input

                for tensor_i in range(steps - 1):
                    rand = np.random.randint(2)
                    # teach from forward as well as from known tensor alternatively
                    if rand == 1:
                        next_tensor, encoder_hidden = self._encoder.forward(
                            batch[:, tensor_i, :].unsqueeze(dim=1),
                            encoder_hidden)
                    else:
                        next_tensor, encoder_hidden = self._encoder.forward(next_tensor.detach(), encoder_hidden)

                    loss += self._criterion(next_tensor, batch[:, tensor_i + 1, :].unsqueeze(dim=1))

                # decode
                decoder_hidden = encoder_hidden
                next_tensor = torch.full_like(next_tensor, self._sos, dtype=torch.float32).to(self.device)
                tensor_target = torch.cat([next_tensor, batch], dim=1)  # add SOS token at t=0 to true input

                for tensor_i in range(steps - 1):
                    rand = np.random.randint(2)
                    # teach from forward as well as from known tensor alternatively
                    if rand == 1:
                        next_tensor, decoder_hidden = self._decoder.forward(
                            tensor_target[:, tensor_i, :].unsqueeze(dim=1),
                            decoder_hidden)
                    else:
                        next_tensor, decoder_hidden = self._decoder.forward(next_tensor.detach(), decoder_hidden)

                    loss += self._criterion(next_tensor, tensor_target[:, tensor_i + 1, :].unsqueeze(dim=1))

                average_loss += loss.item()
                loss.backward()
                self._optimizer.step()

            average_loss = average_loss / len(priming_data)

            if average_loss < self._stop_on_error:
                break
            if feedback_hoop_function is not None:
                feedback_hoop_function("epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(
                    epoch_n=i + 1,
                    total=self._train_iters,
                    average_loss=average_loss))

        self._prepared = True

    def _encode_one(self, data, previous=None, initial_hidden=None, return_next_value=False):
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
            # n_timesteps inferred from query
            data_tensor = tensor_from_series(data, self.device, self._n_dims,
                                             self._eos, normalizer=self._normalizer)

            if previous is not None:
                target_tensor = torch.Tensor(previous).to(self.device)
                target_tensor[torch.isnan(target_tensor)] = 0.0
                if len(target_tensor.shape) < 3:
                    target_tensor = target_tensor.transpose(0, 1).unsqueeze(0)
                data_tensor = torch.cat((data_tensor, target_tensor), dim=-1)

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

    def encode(self, column_data, previous_target_data=None, get_next_count=None):
        """
        Encode a list of time series data
        :param column_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...] to encode
        :param get_next_count: default None, but you can pass a number X and it will return the X following predictions
                               on the series for each ts_data_point in column_data
        :return: a list of encoded time series or if get_next_count !=0 two lists (encoded_values, projected_numbers)
        """

        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        for i in range(len(column_data)):
            if not isinstance(column_data[i][0], list):
                column_data[i] = [column_data[i]]  # add dimension for 1D timeseries

        # include autoregressive target data
        ptd = []
        if previous_target_data is not None and len(previous_target_data) > 0:
            for i, col in enumerate(previous_target_data):
                normalizer = self._target_ar_normalizers[i]
                ptd.append(normalizer.encode(col))

        ret = []
        next = []

        for i, val in enumerate(column_data):
            if get_next_count is None:
                if previous_target_data is not None and len(previous_target_data) > 0:
                    encoded = self._encode_one(val, previous=[values[i] for values in ptd])
                else:
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
                    next_i.append(next_reading)

                next_value = next_i[0][0].cpu()

                if self._normalizer:
                    next_value = torch.Tensor(self._normalizer.inverse_transform(next_value))

                next.append(next_value)

            ret.append(encoded[0][0].cpu())

        if get_next_count is None:
            return self._pytorch_wrapper(torch.stack(ret))
        else:
            return self._pytorch_wrapper(torch.stack(ret)), self._pytorch_wrapper(torch.stack(next))

    def _decode_one(self, hidden, steps):
        """
        Decodes a single time series from its encoded representation.
        :param hidden: time series embedded representation tensor, with size self._encoded_vector_size
        :param steps: as in decode(), defines how many values to output when reconstructing
        :return: decoded time series list
        """
        self._decoder.eval()
        with torch.no_grad():
            ret = []
            next_tensor = torch.full((1, 1, self._n_dims), self._sos, dtype=torch.float32).to(self.device)
            timesteps = steps if steps else self._max_ts_length
            for _ in range(timesteps):
                next_tensor, hidden = self._decoder.forward(next_tensor, hidden)
                ret.append(next_tensor)
            return torch.stack(ret)

    def decode(self, encoded_data, steps=None):
        """
        Decode a list of embedded multidimensional time series
        :param encoded_data: a list of embeddings [ e1, e2, ...] to be decoded into time series
        :param steps: fixed number of timesteps to reconstruct from each embedding.
        If None, encoder will output the largest length encountered during training.
        :return: a list of reconstructed time series
        """
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        for _, val in enumerate(encoded_data):
            hidden = torch.unsqueeze(torch.unsqueeze(val, dim=0), dim=0).to(self.device)
            reconstruction = self._decode_one(hidden, steps).cpu().squeeze().T.numpy()

            if self._n_dims == 1:
                reconstruction = reconstruction.reshape(1, -1)

            if self._normalizer:
                reconstruction = self._normalizer.inverse_transform(reconstruction)

            ret.append(reconstruction)

        return self._pytorch_wrapper(ret)
