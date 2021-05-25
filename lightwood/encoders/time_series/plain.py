import torch
import numpy as np
from itertools import product

from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.encoders.time_series.helpers.common import MinMaxNormalizer, CatNormalizer, get_group_matches


class TimeSeriesPlainEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        """
        Fits a normalizer for a time series previous historical data.
        When encoding, it returns a normalized window of previous data.
        """
        super().__init__(is_target)
        self.original_type = None
        self._normalizers = {}

    def prepare(self, priming_data, previous_target_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        for group_name, norm in previous_target_data[0]['normalizers'].items():
            self._normalizers[group_name] = norm

        self._prepared = True

    def encode(self, column_data, group_info):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        raw_data = np.array(column_data)
        data = torch.cat([self._normalizers['__default'].encode(raw_data)], dim=-1)  # refined with group info

        for combination in list(product(*[set(x) for x in group_info[0]['group_info'].values()])):
            combination = frozenset(combination)
            idxs, subset = get_group_matches(group_info[0], combination, group_info[0]['group_info'].keys())
            if subset.size > 0:
                data[idxs] =  self._normalizers[combination].encode(subset)

        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0
        return data
