import math
import sys

import torch
import numpy as np
from lightwood.encoder import BaseEncoder
from lightwood.encoder.numeric import TsNumericEncoder
from lightwood.helpers.log import log


class TsArrayNumericEncoder(BaseEncoder):
    """
    Variant of vanilla numerical encoder, supports dynamic mean re-scaling
    """
    def __init__(self, timesteps, is_target=False, grouped_by=None):
        super(TsArrayNumericEncoder, self).__init__(is_target=is_target)
        # time series normalization params
        self.normalizers = None
        self.sub_encoders = [TsNumericEncoder(is_target=is_target, grouped_by=grouped_by) for _ in range(timesteps)]
        self.group_combinations = None
        self.dependencies = grouped_by

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        for encoder in self.sub_encoders:
            encoder.prepare(priming_data)

        self._prepared = True

    def encode(self, data, dependency_data={}):
        """dependency_data: dict with grouped_by column info,
        to retrieve the correct normalizer for each datum"""
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if not dependency_data:
            dependency_data = {'__default': [None] * len(data)}

        ret = []
        for encoder in self.sub_encoders:
            ret.append(encoder.encode(data, dependency_data=dependency_data))

        return torch.stack(ret)

    def decode(self, encoded_values, dependency_data=None):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        for encoder in self.sub_encoders:
            ret.append(encoder.decode(encoded_values, dependency_data=dependency_data))

        return ret
