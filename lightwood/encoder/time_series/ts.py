from typing import List, Iterable

import torch

from type_infer.dtype import dtype
from lightwood.encoder.array import ArrayEncoder


class TimeSeriesEncoder(ArrayEncoder):
    is_timeseries_encoder: bool = True
    is_trainable_encoder: bool = True

    def __init__(self, stop_after: float, window: int = None, is_target: bool = False, original_type: dtype = None):
        """
        Time series encoder. This module will pass the normalized series values, along with moving averages taken from the series' last `window` values.
        :param stop_after: time budget in seconds.
        :param window: expected length of array data.
        :param original_type: element-wise data type
        """  # noqa
        super().__init__(stop_after, window, is_target, original_type)
        self.max_mavg_offset = self.output_size
        self.output_size += self.max_mavg_offset

    def encode(self, column_data: Iterable[Iterable]) -> torch.Tensor:
        """
        Encodes time series data.

        :param column_data: Input column data to be encoded
        :returns: a torch tensor representing the encoded time series.
        """

        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        base_encode = super().encode(column_data)

        if self.original_type in (dtype.integer, dtype.float, dtype.quantity, dtype.num_tsarray):
            mavgs = []
            for offset in range(self.max_mavg_offset):
                ma = torch.mean(base_encode[:, offset:self.max_mavg_offset], 1)
                mavgs.append(ma.unsqueeze(1))
            base_encode[:, (self.output_size - self.max_mavg_offset):] = torch.cat(mavgs, dim=1)

        return base_encode

    def decode(self, data: torch.Tensor) -> List[Iterable]:
        """
        Converts data as a list of arrays. Removes all encoded moving average information.

        :param data: Encoded data prepared by this array encoder
        :returns: A list of iterable sequences in the original data space
        """
        decoded = data[:, self.max_mavg_offset].tolist()
        return decoded
