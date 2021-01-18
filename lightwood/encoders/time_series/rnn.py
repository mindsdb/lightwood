from lightwood.encoders.time_series.helpers.rnn_helpers import *
from lightwood.encoders.time_series.helpers.transformer_helpers import *
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.datetime import DatetimeEncoder
from lightwood.helpers.device import get_devices

import torch
import torch.nn as nn
from torch import optim
from math import gcd


class TimeSeriesEncoder(BaseEncoder):

    def __init__(self, encoded_vector_size=128, train_iters=100, stop_on_error=0.01, learning_rate=0.01,
                 is_target=False, ts_n_dims=1, encoder_class=EncoderRNNNumerical):
        super().__init__(is_target)
        self.device, _ = get_devices()
        self.encoder_class = encoder_class
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._transformer_hidden_size = None
        self._epochs = train_iters  # training epochs
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self._is_setup = False
        self._max_ts_length = 0
        self._sos = 0.0  # start of sequence for decoding
        self._eos = 0.0  # end of input sequence -- padding value for batches
        self._n_dims = ts_n_dims
        self._normalizer = None
        self._target_ar_normalizers = []

    def setup_nn(self, additional_targets=None):
        """This method must be executed after initializing, else types are unassigned"""
        if 'datetime' in (self.original_type, self.secondary_type):
            self._normalizer = DatetimeEncoder(sinusoidal=True)
            self._n_dims *= len(self._normalizer.fields)*2  # sinusoidal datetime components
        elif 'numeric' in (self.original_type, self.secondary_type):
            self._normalizer = MinMaxNormalizer()

        total_dims = self._n_dims
        dec_hsize = self._encoded_vector_size

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

        if self.encoder_class == EncoderRNNNumerical:
            self._enc_criterion = nn.MSELoss()
            self._dec_criterion = self._enc_criterion
            self._encoder = self.encoder_class(input_size=total_dims,
                                               hidden_size=self._encoded_vector_size).to(self.device)

        elif self.encoder_class == TransformerEncoder:
            self._enc_criterion = self._masked_criterion
            self._dec_criterion = nn.MSELoss()
            self._base_criterion = nn.MSELoss(reduction="none")
            if self._transformer_hidden_size is None:
                self._transformer_hidden_size = total_dims*2  # arbitrary

            self._encoder = self.encoder_class(ninp=total_dims,
                                               nhead=gcd(dec_hsize, total_dims),
                                               nhid=self._transformer_hidden_size,
                                               nlayers=1).to(self.device)

        self._decoder = DecoderRNNNumerical(output_size=total_dims, hidden_size=dec_hsize).to(self.device)
        self._parameters = list(self._encoder.parameters()) + list(self._decoder.parameters())
        self._optimizer = optim.AdamW(self._parameters, lr=self._learning_rate, weight_decay=1e-4)
        self._is_setup = True

    def to(self, device, available_devices):
        if self._is_setup:
            self.device = device
            self._encoder = self._encoder.to(self.device)
            self._decoder = self._decoder.to(self.device)
        return self

    def _prepare_raw_data(self, data):
        """Convert to array and determine max length"""
        out_data = []
        for e in data:
            if not isinstance(e, torch.Tensor):
                t = torch.tensor(e, dtype=torch.float)
            else:
                t = e.float()
            t[torch.isnan(t)] = 0.0
            out_data.append(t)
        lengths = torch.tensor([len(e) for e in data], dtype=torch.float)
        return out_data, lengths

    def _get_batch(self, source, start, step):
        # source is an iterable element, we want to get source[i+step] or source[i+end]
        # If padding is not None, until size `source[i+step]`
        end = min(start + step, len(source))
        return source[start:end]

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

        # Convert to array and determine max length
        priming_data, lengths_data = self._prepare_raw_data(priming_data)
        self._max_ts_length = int(lengths_data.max())

        if self._normalizer:
            self._normalizer.prepare(priming_data)
            priming_data = torch.stack([self._normalizer.encode(d) for d in priming_data]).to(self.device)
        else:
            priming_data = torch.stack([d for d in priming_data]).unsqueeze(-1).to(self.device)

        # merge all normalized data into a training batch
        if previous_target_data is not None and len(previous_target_data) > 0:
            normalized_tensors = []
            for target_dict in previous_target_data:
                normalizer = target_dict['normalizer']
                self._target_ar_normalizers.append(normalizer)
                data = torch.Tensor(normalizer.encode(target_dict['data'])).to(self.device)
                data[torch.isnan(data)] = 0.0
                if len(data.shape) < 3:
                    data = data.unsqueeze(-1)  # add feature dimension
                normalized_tensors.append(data)

            normalized_data = torch.cat(normalized_tensors, dim=-1)
            priming_data = torch.cat([priming_data, normalized_data], dim=-1)

        self._encoder.train()
        for i in range(self._epochs):
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
                        len_batch = self._get_batch(lengths_data, batch_idx, min(batch_idx + batch_size, len(priming_data)))
                        batch = batch, len_batch

                        next_tensor, hidden_state, dec_loss = self._encoder.bptt(batch, self._enc_criterion, self.device)
                        loss += dec_loss

                    else:
                        next_tensor, hidden_state, enc_loss = self._encoder.bptt(batch, self._enc_criterion, self.device)
                        loss += enc_loss

                        next_tensor, hidden_state, dec_loss = self._decoder.decode(batch, next_tensor, self._dec_criterion,
                                                                                   self.device,
                                                                                   hidden_state=hidden_state)
                        loss += dec_loss

                loss.backward()

                self._optimizer.step()
                average_loss += loss.item()

            average_loss = average_loss / len(priming_data)
            batch_idx += batch_size

            if average_loss < self._stop_on_error:
                break
            if feedback_hoop_function is not None:
                feedback_hoop_function("epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(
                    epoch_n=i + 1,
                    total=self._epochs,
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
            # Convert to array and determine max length
            data, lengths_data = self._prepare_raw_data(data)
            _max_ts_length = int(lengths_data.max())

            if self._normalizer:
                data = torch.stack([self._normalizer.encode(d) for d in data]).to(self.device)
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
