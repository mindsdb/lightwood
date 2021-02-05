from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.datetime import DatetimeEncoder
from lightwood.encoders.time_series.helpers.rnn_helpers import *
import torch


class TimeSeriesPlainEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        """ Very simple encoder.
        This simple encoder fits a normalizer using previous historical
        data, and when encoding it ignores the temporal column entirely,
        instead passing only the normalized window of previous data.
        """
        super().__init__(is_target)
        self._normalizer = None
        self._target_ar_normalizers = []

    def prepare(self, priming_data, previous_target_data=None, feedback_hoop_function=None):
        print("HI")
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        else:
            if previous_target_data:
                for t in previous_target_data:
                    if t['original_type'] == 'categorical':
                        normalizer = CatNormalizer()
                    else:
                        normalizer = MinMaxNormalizer()

                    normalizer.prepare(t['data'])
                    self._target_ar_normalizers.append(normalizer)
            else:
                raise Exception('Plain time series encoder needs previous target data.')
        self._prepared = True

    def encode(self, column_data, previous_target_data=None):
        print("HIHIHI")
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if not previous_target_data:
            raise Exception('Plain time series encoder needs previous target data.')

        data = []
        for i, col in enumerate(previous_target_data):
            normalizer = self._target_ar_normalizers[i]
            data.append(normalizer.encode(col))

        return torch.stack(data)
