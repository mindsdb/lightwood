import math
import sys

import torch
import numpy as np
from lightwood.encoders.numeric import NumericEncoder
from lightwood.logger import log


class TsNumericEncoder(NumericEncoder):
    """
    Variant of vanilla numerical encoder, supports dynamic mean re-scaling
    """
    def __init__(self, is_target=False):
        super(TsNumericEncoder, self).__init__(is_target=is_target)
        # time series normalization params
        self.normalizers = None
        self.group_combinations = None

    def encode(self, data, extra_data=None):
        """extra_data[0]['group_info']: dict with all grouped_by column info,
        to retrieve the correct normalizer for each datum"""
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if extra_data is None or not extra_data[0]['group_info']:
            group_info = {'__default': [None] * len(data)}
        else:
            group_info = extra_data[0]['group_info']

        ret = []
        for real, group in zip(data, list(zip(*group_info.values()))):
            try:
                real = float(real)
            except:
                try:
                    real = float(real.replace(',','.'))
                except:
                    real = None
            if self.is_target:
                vector = [0] * 3
                if group is not None and self.normalizers is not None:
                    try:
                        mean = self.normalizers[frozenset(group)].abs_mean
                    except KeyError:
                        # novel group-by, we use default normalizer mean
                        mean = self.normalizers['__default'].abs_mean
                else:
                    mean = self._abs_mean
                if real is not None and mean > 0:
                    vector[0] = 1 if real < 0 and not self.positive_domain else 0
                    vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                    vector[2] = real / mean
                else:
                    log.debug(f'Can\'t encode target value: {real}')

            else:
                vector = [0] * 4
                try:
                    if real is None:
                        vector[0] = 0
                    else:
                        vector[0] = 1
                        vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                        vector[2] = 1 if real < 0 and not self.positive_domain else 0
                        vector[3] = real/self._abs_mean
                except Exception as e:
                    vector = [0] * 4
                    log.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(vector)

        return torch.Tensor(ret)

    def decode(self, encoded_values, decode_log=None, group_info=None):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        ret = []
        if not group_info:
            group_info = {'__default': [None] * len(encoded_values)}
        if type(encoded_values) != type([]):
            encoded_values = encoded_values.tolist()

        for vector, group in zip(encoded_values, list(zip(*group_info.values()))):
            if self.is_target:
                if np.isnan(vector[0]) or vector[0] == float('inf') or np.isnan(vector[1]) or vector[1] == float('inf') or np.isnan(vector[2]) or vector[2] == float('inf'):
                    log.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10, 63)
                else:
                    if decode_log:
                        sign = -1 if vector[0] > 0.5 else 1
                        try:
                            real_value = math.exp(vector[1]) * sign
                        except OverflowError as e:
                            real_value = pow(10,63) * sign
                    else:
                        if group is not None and self.normalizers is not None:
                            try:
                                mean = self.normalizers[frozenset(group)].abs_mean
                            except KeyError:
                                # decode new group with default normalizer
                                mean = self.normalizers['__default'].abs_mean
                        else:
                            mean = self._abs_mean

                        real_value = vector[2] * mean

                    if self.positive_domain:
                        real_value = abs(real_value)

                    if self._type == 'int':
                        real_value = int(real_value)

            else:
                if vector[0] < 0.5:
                    ret.append(None)
                    continue

                real_value = vector[3] * self._abs_mean

                if self._type == 'int':
                    real_value = round(real_value)

            ret.append(real_value)
        return ret
