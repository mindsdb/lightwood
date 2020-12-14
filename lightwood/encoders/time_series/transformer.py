import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.time_series.helpers.transformer_helpers import (
    TransformerModel,
    len_to_mask,
)


class TransformerEncoder(BaseEncoder):
    def __init__(
        self,
        encoded_vector_size=4,
        train_iters=100,
        stop_on_error=0.01,
        learning_rate=0.01,
        is_target=False,
    ):
        super().__init__(is_target)
        self.device, _ = get_devices()

        # Model. We use encoded_vector_size for input and hidden
        self._encoder = TransformerModel(
            ninp=encoded_vector_size, nhead=2, nhid=encoded_vector_size, nlayers=2
        ).to(self.device)

        # Training params
        self._train_iters = train_iters  # epochs
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate

        # It would be worth to consider the use of a scheduler with a warm-up in some datasets
        self._optimizer = optim.AdamW(
            self._encoder.parameters(), lr=self._learning_rate, weight_decay=1e-4
        )
        self._criterion = nn.MSELoss(reduction="none")

        self._prepared = False
        self.bptt = 35
        self.gradient_norm_clip = 0.5

        # Lezcano: These should be global constants of the library
        self._sos = 0.0  # start of sequence for decoding
        self._eos = 0.0  # end of input sequence -- padding value for batches

    def _append_eos(self, data):
        for i in range(len(data)):
            data[i].append(self._eos)

    def to(self, device, available_devices):
        self.device = device
        self._encoder = self._encoder.to(self.device)
        return self

    def _get_batch(self, source, start, step):
        # source is an iterable element, we want to get source[i+step] or source[i+end]
        # If padding is not None, until size `source[i+step]`
        end = min(start + step, len(source))
        return source[start:end]

    def _get_chunk(self, source, source_lengths, start, step):
        end = min(start + step, len(source) - 1)
        # Compute the lenghts of the sequences
        # The -1 comes from the fact that the last element is used as target but not as data!
        lengths = torch.clamp((source_lengths - 1) - start, min=0, max=step)
        # This is necessary for MultiHeadedAttention to work
        non_empty = lengths != 0
        data = source[start:end]
        target = source[start + 1 : end + 1]
        data, target, lengths = (
            data[:, non_empty],
            target[:, non_empty],
            lengths[non_empty],
        )
        return data, target, lengths

    def _masked_criterion(self, output, targets, lengths):
        """ Computes the loss of the first `lengths` items in the chunk """
        mask = len_to_mask(lengths, zeros=False)
        # Put in (B, T) format and zero-out the unnecessary values
        output = output.t() * mask
        targets = targets.t() * mask
        # compute the loss with respect to the appropriate lengths and average across the batch-size
        # We compute for every output (x_i)_i=1^L and target (y_i)_i=1^L, loss = 1/L \sum (x_i - y_i)^2
        # And average across the mini-batch
        losses = self._criterion(output, targets).sum(dim=1)
        # The TBPTT will compute a slightly different loss, but it is not problematic
        loss = torch.dot((1.0 / lengths.float()), losses) / len(losses)
        return loss

    def _prepare_raw_data(self, data):
        # Convert to array and determine max length:
        # priming_data is a list of len B with lists of length in [0, T]
        # If the length of the series varies a lot, it would be worth it to order them and apply a few optimisations
        # we do not do so now
        out_data = [
            torch.tensor(e, dtype=torch.float, device=self.device) for e in data
        ]
        lengths = torch.tensor(
            [len(e) for e in data], dtype=torch.long, device=self.device
        )
        return out_data, lengths

    def prepare_encoder(
        self, priming_data, feedback_hoop_function=None, batch_size=256
    ):
        """
        The usual, run this on the initial training data for the encoder
        :param priming_data: a list of 1-dimensional time series [[list1], ..., [listT]]
        :param feedback_hoop_function: [if you want to get feedback on the training process]
        :param batch_size: int
        :return:
        """
        if self._prepared:
            raise RuntimeError(
                'You can only call "prepare_encoder" once for a given encoder.'
            )

        # We append the eos symbol
        self._append_eos(priming_data)
        priming_data, lengths_data = self._prepare_raw_data(priming_data)
        npoints = len(priming_data)

        # We pad the sequence with the function from PyTorch
        self._encoder.train()
        for epoch in range(self._train_iters):
            total_loss = 0.0
            for start_batch in range(0, npoints, batch_size):
                self._optimizer.zero_grad()

                train_batch = self._get_batch(priming_data, start_batch, batch_size)
                len_batch = self._get_batch(lengths_data, start_batch, batch_size)
                train_batch = pad_sequence(train_batch)

                # TBPTT
                losses = []
                with LightwoodAutocast():
                    for start_chunk in range(0, train_batch.size(0) - 1, self.bptt):
                        data, targets, lengths_chunk = self._get_chunk(
                            train_batch, len_batch, start_chunk, self.bptt
                        )
                        data = data.unsqueeze(-1)
                        output = self._encoder(data, lengths_chunk)
                        output = output.squeeze(-1)
                        losses.append(
                            self._masked_criterion(output, targets, lengths_chunk)
                        )
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._encoder.parameters(), self.gradient_norm_clip)
                self._optimizer.step()

                total_loss += loss.item() * len(train_batch)

            average_loss = total_loss / npoints
            if average_loss < self._stop_on_error:
                break

            feedback_hoop_function(
                "epoch [{epoch_n}/{total}] average_loss = {average_loss}".format(
                    epoch_n=epoch, total=self._train_iters, average_loss=average_loss
                )
            )
        self._prepared = True

    def encode(self, column_data):
        """
        Encode a list of time series data
        :param column_data: a list of 1-dimensional time series [[list1], ..., [listT]] to encode
        :return: a list of encoded time series
        """
        if not self._prepared:
            raise RuntimeError(
                'You need to call "prepare_encoder" before calling "encode".'
            )
        self._encoder.eval()

        # The functionality get_next_count is not implementd, as implementing a decoder is not part of the task
        # We assume that the whole data is not "too long" (i.e. no batching for now)

        column_data, lengths_data = self._prepare_raw_data(column_data)
        column_data = pad_sequence(column_data).unsqueeze(-1)
        out = self._encoder(column_data, lengths_data)
        out = out.squeeze(-1).t()
        out = list(torch.unbind(out))
        # Remove the padding
        for i, (d, l) in enumerate(zip(out, lengths_data)):
            out[i] = list(d[:l])
        return out
