import time
from math import gcd
from typing import List
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.encoder.datetime import DatetimeNormalizerEncoder
from lightwood.encoder.time_series.helpers.rnn_helpers import EncoderRNNNumerical, DecoderRNNNumerical
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
from lightwood.helpers.ts import get_group_matches
from lightwood.encoder.time_series.helpers.transformer_helpers import TransformerEncoder, get_chunk, len_to_mask


class TimeSeriesEncoder(BaseEncoder):
    """
    Time series encoder. This module can learn features for any `order_by` temporal column, both with and without accompanying target data.

    The backbone of this encoder is either a recurrent neural network or a transformer; both structured in an encoder-decoder fashion.
    """  # noqa
    is_timeseries_encoder: bool = True
    is_trainable_encoder: bool = True

    def __init__(self, stop_after: float, is_target=False, original_type: str = None, target: str = None,
                 grouped_by: List[str] = [], encoder_type='rnn'):
        super().__init__(is_target)
        self.device, _ = get_devices()
        self.target = target
        self.grouped_by = grouped_by
        self._learning_rate = 0.01
        self.output_size = 128
        self._transformer_hidden_size = None
        self._epochs = int(1e5)  # default training epochs
        self._stop_on_n_bad_epochs = 5  # stop training after N epochs where loss is worse than running avg
        self._epochs_running_avg = 5  # amount of epochs for running average
        self._pytorch_wrapper = torch.FloatTensor
        self.is_prepared = False
        self._is_setup = False
        self._max_ts_length = 0
        self._sos = 0.0  # start of sequence for decoding
        self._eos = 0.0  # end of input sequence -- padding value for batches
        self._n_dims = 1
        self._normalizer = None
        self.dep_norms = {}  # dict of dict of normalizers for each dependency (can be grouped-by some column)
        self._target_type = None
        self._group_combinations = None
        self.original_type = original_type
        self.stop_after = stop_after
        if encoder_type.lower() == 'rnn':
            self.encoder_class = EncoderRNNNumerical
        elif encoder_type.lower() == 'transformer':
            self.encoder_class = TransformerEncoder

    def setup_nn(self, ts_analysis, dependencies=None):
        """This method must be executed after initializing, else types are unassigned"""
        if self.original_type in (dtype.datetime, dtype.date):
            self._normalizer = DatetimeNormalizerEncoder(sinusoidal=True)
            self._n_dims *= len(self._normalizer.fields) * 2  # sinusoidal datetime components
        elif self.original_type in (dtype.float, dtype.integer):
            self._normalizer = MinMaxNormalizer()

        total_dims = self._n_dims
        dec_hsize = self.output_size

        if dependencies:
            for dep_name, dep in dependencies.items():
                self.dependencies.append(dep_name)

                if dep_name in self.grouped_by:
                    continue  # we only use group column for indexing and selecting rows

                assert dep['original_type'] in (dtype.categorical, dtype.binary, dtype.cat_tsarray,
                                                dtype.integer, dtype.float, dtype.num_tsarray)

                if f'__mdb_ts_previous_{self.target}' == dep_name:
                    self.dep_norms[dep_name] = ts_analysis['target_normalizers']
                    self._group_combinations = ts_analysis['group_combinations']
                    self._target_type = dep['original_type']

                # if TS analysis yields no normalizers for this dependency, we create a generic one based on its dtype
                else:
                    if dep['original_type'] in (dtype.categorical, dtype.binary):
                        self.dep_norms[dep_name]['__default'] = CatNormalizer()
                    else:
                        self.dep_norms[dep_name]['__default'] = MinMaxNormalizer()

                    self.dep_norms[dep_name]['__default'].prepare(dep['data'])
                    self._group_combinations = {'__default': None}

                # add descriptor size to the total encoder output dimensionality
                if dep['original_type'] in (dtype.categorical, dtype.binary):
                    total_dims += len(self.dep_norms[dep_name]['__default'].scaler.categories_[0])
                elif dep['original_type'] in (dtype.integer, dtype.float, dtype.num_tsarray, dtype.cat_tsarray):
                    total_dims += 1

        if self.encoder_class == EncoderRNNNumerical:
            self._enc_criterion = nn.MSELoss()
            self._dec_criterion = self._enc_criterion
            self._encoder = self.encoder_class(input_size=total_dims,
                                               hidden_size=self.output_size).to(self.device)
        elif self.encoder_class == TransformerEncoder:
            self._enc_criterion = self._masked_criterion
            self._dec_criterion = nn.MSELoss()
            self._base_criterion = nn.MSELoss(reduction="none")
            if self._transformer_hidden_size is None:
                self._transformer_hidden_size = total_dims * 2  # arbitrary

            self._encoder = self.encoder_class(ninp=total_dims,
                                               nhead=gcd(dec_hsize, total_dims),
                                               nhid=self._transformer_hidden_size,
                                               nlayers=1).to(self.device)
        else:
            raise Exception(f"Time series encoder class not supported: {self.encoder_class}")

        self._decoder = DecoderRNNNumerical(output_size=total_dims, hidden_size=dec_hsize).to(self.device)
        self._parameters = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self._optimizer = optim.AdamW(self._parameters, lr=self._learning_rate, weight_decay=1e-4)
        self._n_dims = total_dims
        self._is_setup = True

    def to(self, device, available_devices):
        if self._is_setup:
            self.device = device
            return super().to(device, available_devices)
        return self

    def _prepare_raw_data(self, data):
        """Convert to array and determine max length"""
        out_data = []
        for e in data:
            if not isinstance(e, torch.Tensor):
                e = np.array(e, dtype=float)
                e[np.isnan(e)] = 0.0
                t = torch.tensor(e, dtype=torch.float)
            else:
                t = e.float()
            t[torch.isnan(t)] = 0.0
            out_data.append(t)
        lengths = torch.tensor([len(e) for e in data], dtype=torch.float)
        return out_data, lengths

    def _get_batch(self, source, start, end):
        end = min(end, len(source))
        return source[start:end]

    def prepare(self, train_priming_data: pd.Series, dev_priming_data: pd.Series, dependency_data={}, ts_analysis=None,
                feedback_hoop_function=log.info, batch_size=256):
        """
        :param priming_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...]
        :param dependency_data: raw data from other columns
        :param ts_analysis: dictionary with time analysis info (e.g. normalizers for each target group)
        :param feedback_hoop_function: method to use if you want to get feedback on the training process
        :param batch_size
        """
        priming_data = pd.concat([train_priming_data, dev_priming_data])
        priming_data = list(priming_data.values)

        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        else:
            self.setup_nn(ts_analysis, dependency_data)

        started = time.time()

        # Convert to array and determine max length
        priming_data, lengths_data = self._prepare_raw_data(priming_data)
        self._max_ts_length = int(lengths_data.max())

        if self._normalizer:
            self._normalizer.prepare(priming_data)
            priming_data = self._normalizer.encode(priming_data).to(self.device)
            if len(priming_data.shape) < 3:
                priming_data = priming_data.unsqueeze(-1)
        else:
            priming_data = torch.stack([d for d in priming_data]).unsqueeze(-1).to(self.device)

        # merge all normalized data into a training batch
        normalized_tensors = []
        for dep_name, dep_data in dependency_data.items():
            if dep_name in self.grouped_by:
                continue
            if dep_data['original_type'] in (dtype.integer, dtype.float):
                dep_data['group_info'] = {group: dependency_data[group]['data'] for group in self.grouped_by}
                data = torch.zeros((len(priming_data), lengths_data.max().int().item(), 1))
                all_idxs = set(range(len(data)))
                for group_name, normalizer in self.dep_norms[dep_name].items():
                    if group_name != '__default':
                        idxs, subset = get_group_matches(dep_data, normalizer.combination)
                        normalized = normalizer.encode(subset).unsqueeze(-1)
                        data[idxs, :, :] = normalized
                        all_idxs -= set(idxs)
                if len(all_idxs) > 0 and '__default' in self.dep_norms[dep_name].keys():
                    default_norm = self.dep_norms[dep_name]['__default']
                    subset = [dep_data['data'][idx] for idx in list(all_idxs)]
                    data[list(all_idxs), :, :] = torch.Tensor(default_norm.encode(subset)).unsqueeze(-1)

            else:
                # categorical has only one normalizer at all times
                normalizer = self.dep_norms[dep_name]['__default']
                data = normalizer.encode(dep_data['data'].values)
                if len(data.shape) < 3:
                    data = data.unsqueeze(-1)  # add feature dimension
            data[torch.isnan(data)] = 0.0
            normalized_tensors.append(data)

        if normalized_tensors:
            normalized_data = torch.cat(normalized_tensors, dim=-1).to(self.device)
            priming_data = torch.cat([priming_data, normalized_data], dim=-1)

        self._encoder.train()
        running_losses = np.full(self._epochs_running_avg, np.nan)
        bad_epochs = 0

        for epoch in range(self._epochs):
            average_loss = 0

            for batch_idx in range(0, len(priming_data), batch_size):
                # setup loss and optimizer
                self._optimizer.zero_grad()
                loss = 0

                # shape: (batch_size, timesteps, n_dims)
                batch = self._get_batch(priming_data, batch_idx, min(batch_idx + batch_size, len(priming_data)))

                # encode and decode through time
                with LightwoodAutocast():
                    if self.encoder_class == TransformerEncoder:
                        # pack batch length info tensor
                        len_batch = self._get_batch(lengths_data, batch_idx, min(
                            batch_idx + batch_size, len(priming_data)))
                        batch = batch, len_batch

                        next_tensor, hidden_state, dec_loss = self._encoder.bptt(
                            batch, self._enc_criterion, self.device)
                        loss += dec_loss

                    else:
                        next_tensor, hidden_state, enc_loss = self._encoder.bptt(
                            batch, self._enc_criterion, self.device)
                        loss += enc_loss

                        next_tensor, hidden_state, dec_loss = self._decoder.decode(
                            batch, next_tensor, self._dec_criterion, self.device, hidden_state=hidden_state)
                        loss += dec_loss

                loss.backward()

                self._optimizer.step()
                average_loss += loss.item()

            average_loss = average_loss / len(priming_data)
            batch_idx += batch_size

            if epoch > self._epochs_running_avg and average_loss > np.average(running_losses):
                bad_epochs += 1

            # update running loss
            running_losses[:-1] = running_losses[1:]
            running_losses[-1] = average_loss

            if feedback_hoop_function is not None:
                feedback_hoop_function(
                    "time series encoder epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(
                        epoch_n=epoch + 1, total=self._epochs, average_loss=average_loss))

            if bad_epochs > self._stop_on_n_bad_epochs:
                break
            elif (time.time() - started) > self.stop_after:
                break

        self.is_prepared = True

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
            # Convert to array and determine max length
            data, lengths_data = self._prepare_raw_data(data)
            self._max_ts_length = int(lengths_data.max())

            if self._normalizer:
                data = self._normalizer.encode(data).to(self.device)
                if len(data.shape) < 3:
                    data = data.unsqueeze(-1)
            else:
                data = torch.stack([d for d in data]).unsqueeze(-1).to(self.device)

            if previous is not None:
                target_tensor = torch.stack(previous).to(self.device)
                target_tensor[torch.isnan(target_tensor)] = 0.0
                if len(target_tensor.shape) < 3:
                    target_tensor = target_tensor.transpose(0, 1).unsqueeze(0)
                data_tensor = torch.cat((data, target_tensor), dim=-1)
            else:
                data_tensor = data

            steps = data_tensor.shape[1]

            if self.encoder_class == EncoderRNNNumerical:
                encoder_hidden = self._encoder.init_hidden(self.device)
                encoder_hidden = encoder_hidden if initial_hidden is None else initial_hidden

                next_tensor = None
                for tensor_i in range(steps):
                    next_tensor, encoder_hidden = self._encoder.forward(data_tensor[:, tensor_i, :].unsqueeze(dim=0),
                                                                        encoder_hidden)

            else:
                next_tensor = None
                len_batch = self._get_batch(lengths_data, 0, len(data))
                batch_size, timesteps, _ = data_tensor.shape

                for start_chunk in range(0, timesteps, timesteps):
                    data, targets, lengths_chunk = get_chunk(data_tensor, len_batch, start_chunk, timesteps)
                    data = data.transpose(0, 1)
                    next_tensor, encoder_hidden = self._encoder.forward(data, lengths_chunk, self.device)

        if return_next_value:
            return encoder_hidden, next_tensor
        else:
            return encoder_hidden

    def encode(self, column_data, dependency_data=None, get_next_count=None):
        """
        Encode a list of time series data
        :param column_data: a list of (self._n_dims)-dimensional time series [[dim1_data], ...] to encode
        :param get_next_count: default None, but you can pass a number X and it will return the X following predictions
                               on the series for each ts_data_point in column_data
        :return: a list of encoded time series or if get_next_count !=0 two lists (encoded_values, projected_numbers)
        """

        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(column_data, pd.Series):
            data = deepcopy(column_data.values)  # get a copy to avoid modifying the actual data frame
        else:
            data = column_data

        for i in range(len(data)):
            if not isinstance(data[i][0], list):
                data[i] = [data[i]]  # add dimension for 1D timeseries

        # include autoregressive target data
        ptd = []
        if dependency_data is not None:
            for dep, dep_data in dependency_data.items():
                if dep in self.grouped_by:
                    continue
                # normalize numerical target per group-by
                if self._target_type in (dtype.integer, dtype.float, dtype.num_tsarray):
                    dep_info = {
                        'group_info': {group: dependency_data[group] for group in self.grouped_by},
                        'data': dep_data
                    }
                    tensor = torch.zeros((len(dep_data), len(dep_data[0]), 1)).to(self.device)
                    all_idxs = set(range(len(dep_data)))

                    for combination in [c for c in self._group_combinations if c != '__default']:
                        normalizer = self.dep_norms[dep].get(tuple(combination), None)
                        if normalizer is None:
                            normalizer = self.dep_norms[dep]['__default']
                        idxs, subset = get_group_matches(dep_info, normalizer.combination)
                        if idxs:
                            tensor[idxs, :, :] = torch.Tensor(normalizer.encode(subset)).unsqueeze(-1).to(self.device)
                            all_idxs -= set(idxs)

                    # encode all remaining rows (not belonging to any grouped combination) with default normalizer
                    if all_idxs:
                        default_norm = self.dep_norms[dep]['__default']
                        subset = [dep_data[idx] for idx in all_idxs]
                        tensor[list(all_idxs), :, :] = torch.Tensor(
                            default_norm.encode(subset)).unsqueeze(-1).to(self.device)
                        tensor[torch.isnan(tensor)] = 0.0

                # normalize categorical target
                else:
                    normalizer = self.dep_norms[dep]['__default']
                    tensor = normalizer.encode(dep_data)
                    tensor[torch.isnan(tensor)] = 0.0

                ptd.append(tensor)

        ret = []
        next = []

        for i, val in enumerate(data):
            if get_next_count is None:
                if dependency_data is not None and len(dependency_data) > 0 and len(ptd) > 0:
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
                    next_value = torch.Tensor(self._normalizer.decode(next_value))

                next.append(next_value)

            ret.append(encoded[0][0].cpu())

        if get_next_count is None:
            return torch.stack(ret)
        else:
            return torch.stack(ret), torch.stack(next)

    def _decode_one(self, hidden, steps):
        """
        Decodes a single time series from its encoded representation.
        :param hidden: time series embedded representation tensor, with size self.output_size
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
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        for _, val in enumerate(encoded_data):
            hidden = torch.unsqueeze(torch.unsqueeze(val, dim=0), dim=0).to(self.device)
            reconstruction = self._decode_one(hidden, steps).cpu().squeeze().T.numpy()

            if self._n_dims == 1:
                reconstruction = reconstruction.reshape(1, -1)

            if self._normalizer:
                reconstruction = self._normalizer.decode(reconstruction)

            ret.append(reconstruction)

        return torch.Tensor(ret)

    def _masked_criterion(self, output, targets, lengths):
        """ Computes the loss of the first `lengths` items in the chunk """
        # Put in (B, T) format and zero-out the unnecessary values
        mask = len_to_mask(lengths, zeros=False).t()

        # Inflate to feature dimension
        mask = mask.unsqueeze(-1).repeat(1, 1, output.shape[-1])
        output = output * mask
        targets = targets * mask

        # compute the loss with respect to the appropriate lengths and average across the batch-size
        # We compute for every output (x_i)_i=1^L and target (y_i)_i=1^L, loss = 1/L \sum (x_i - y_i)^2
        # And average across the mini-batch
        losses = self._base_criterion(output, targets).sum(dim=2).sum(dim=0)

        # The TBPTT will compute a slightly different loss, but it is not problematic
        loss = torch.dot((1.0 / lengths.float()), losses) / len(losses)

        return loss
