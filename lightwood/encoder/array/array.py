from typing import Union
import torch
import pandas as pd
import numpy as np
from lightwood.encoder.base import BaseEncoder
from lightwood.api import dtype
from lightwood.encoder.time_series.helpers.common import MinMaxNormalizer, CatNormalizer


class ArrayEncoder(BaseEncoder):
    is_trainable_encoder: bool = True

    def __init__(self, stop_after: int, window: int = None, is_target: bool = False, original_type: dtype = None):
        """
        Fits a normalizer for a time series previous historical data.
        When encoding, it returns a normalized window of previous data.
        """
        super().__init__(is_target)
        self.stop_after = stop_after
        self.original_type = original_type
        self._normalizer = None
        if window is not None:
            self.output_size = window + 1
        else:
            self.output_size = None

    def prepare(self, train_priming_data, dev_priming_data):
        priming_data = pd.concat([train_priming_data, dev_priming_data])
        priming_data = priming_data.values

        if self.output_size is None:
            self.output_size = np.max([len(x) for x in priming_data if x is not None])
        for i in range(len(priming_data)):
            if priming_data[i] is None:
                priming_data[i] = [0] * self.output_size

        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if self.original_type in (dtype.categorical, dtype.binary):
            self._normalizer = CatNormalizer(encoder_class='ordinal')
        else:
            self._normalizer = MinMaxNormalizer()

        if isinstance(priming_data, pd.Series):
            priming_data = priming_data.values

        self._normalizer.prepare(priming_data)
        self.output_size *= self._normalizer.output_size
        self._prepared = True

    def encode(self, column_data: Union[list, np.ndarray]) -> torch.Tensor:
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(column_data, pd.Series):
            column_data = column_data.values

        for i in range(len(column_data)):
            if column_data[i] is None:
                column_data[i] = [0] * self.output_size

        data = torch.cat([self._normalizer.encode(column_data)], dim=-1)
        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0
        return data
