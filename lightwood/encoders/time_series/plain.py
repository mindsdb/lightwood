from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.time_series.helpers.common import *
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
import torch


class TimeSeriesPlainEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        """
        This simple encoder fits a normalizer using previous historical data,
        and when encoding it simply return the normalized window of previous data.
        """
        super().__init__(is_target)
        self.original_type = None
        self.secondary_type = None
        self._normalizer = None
        self.window_size = None

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        else:
            if self.original_type == COLUMN_DATA_TYPES.CATEGORICAL:
                self._normalizer = CatNormalizer()
            else:
                self._normalizer = MinMaxNormalizer()
            self._normalizer.prepare(priming_data)
        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        data = torch.cat([self._normalizer.encode(column_data)], dim=-1)
        data[torch.isnan(data)] = 0.0
        return data
