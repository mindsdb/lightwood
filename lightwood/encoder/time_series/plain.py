from typing import Union

import torch
import pandas as pd
import numpy as np

from lightwood.encoder.base import BaseEncoder
from lightwood.api import dtype
from lightwood.encoder.time_series.helpers.common import MinMaxNormalizer, CatNormalizer


class TimeSeriesPlainEncoder(BaseEncoder):
    def __init__(self, stop_after: int, window: int, is_target=False, original_type=None):
        """
        Fits a normalizer for a time series previous historical data.
        When encoding, it returns a normalized window of previous data.
        """
        super().__init__(is_target)
        self.stop_after = stop_after
        self.original_type = original_type
        self._normalizer = None
        self.output_size = window+1

    def prepare(self, priming_data):
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
        data = torch.cat([self._normalizer.encode(column_data)], dim=-1)
        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0
        return data
