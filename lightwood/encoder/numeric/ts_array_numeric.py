import math
import sys

import torch
import numpy as np
import torch.nn.functional as F
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
        self.sub_encoder = TsNumericEncoder(is_target=is_target, grouped_by=grouped_by)
        self.group_combinations = None
        self.dependencies = grouped_by
        self.data_window = timesteps
        self.out_features = self.data_window*self.sub_encoder.out_features

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.sub_encoder.prepare(priming_data)
        self._prepared = True

    def encode(self, data, dependency_data={}):
        """dependency_data: dict with grouped_by column info,
        to retrieve the correct normalizer for each datum
        :return tensor with shape (batch, NxK) where N: self.data_window and K: sub-encoder # of output features"""
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if not dependency_data:
            dependency_data = {'__default': [None] * len(data)}

        ret = []
        for data_point in data:
            ret.append(self.sub_encoder.encode([data_point], dependency_data=dependency_data))

        ret = torch.hstack(ret)
        padding_size = self.out_features - ret.shape[-1]

        if padding_size > 0:
            ret = F.pad(ret, (0, padding_size))

        return ret

    def decode(self, encoded_values, dependency_data=None, return_all=False):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        encoded_values = encoded_values.reshape(encoded_values.shape[0],
                                                self.data_window,
                                                self.sub_encoder.out_features)

        ret = []
        for encoded_timestep in torch.split(encoded_values, 1, dim=1):
            ret.extend(self.sub_encoder.decode(encoded_timestep.squeeze(1), dependency_data=dependency_data))

        return ret # if return_all else ret[0]
