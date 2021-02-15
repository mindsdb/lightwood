import torch

from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.encoders.time_series.helpers.common import MinMaxNormalizer, CatNormalizer


class TimeSeriesPlainEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        """
        Fits a normalizer for a time series previous historical data.
        When encoding, it returns a normalized window of previous data.
        """
        super().__init__(is_target)
        self.original_type = None
        self._normalizer = None

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if self.original_type == COLUMN_DATA_TYPES.CATEGORICAL:
            self._normalizer = CatNormalizer(encoder_class='ordinal')
        else:
            self._normalizer = MinMaxNormalizer()

        self._normalizer.prepare(priming_data)
        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        data = torch.cat([self._normalizer.encode(column_data)], dim=-1)
        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0
        return data
