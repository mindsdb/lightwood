import math
import sys

import torch
import numpy as np
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log


class NumericEncoder(BaseEncoder):

    def __init__(self, data_type=None, is_target=False):
        super().__init__(is_target)
        self._type = data_type
        self._abs_mean = None
        self.positive_domain = False
        self.decode_log = False

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        value_type = 'int'
        for number in priming_data:
            try:
                number = float(number)
            except:
                continue

            if np.isnan(number):
                err = 'Lightwood does not support working with NaN values !'
                log.error(err)
                raise Exception(err)

            if int(number) != number:
                value_type = 'float'

        self._type = value_type if self._type is None else self._type
        non_null_priming_data = [float(str(x).replace(',','.')) for x in priming_data if x is not None]
        self._abs_mean = np.mean(np.abs(non_null_priming_data))
        self._prepared = True

    def encode(self, data):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        for real in data:
            try:
                real = float(real)
            except:
                try:
                    real = float(real.replace(',','.'))
                except:
                    real = None
            if self.is_target:
                vector = [0] * 3
                if real is not None and self._abs_mean > 0:
                    vector[0] = 1 if real < 0 and not self.positive_domain else 0
                    vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                    vector[2] = real / self._abs_mean
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

    def decode(self, encoded_values, decode_log=None):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        ret = []
        if type(encoded_values) != type([]):
            encoded_values = encoded_values.tolist()

        for vector in encoded_values:
            if self.is_target:
                if np.isnan(vector[0]) or vector[0] == float('inf') or np.isnan(vector[1]) or vector[1] == float('inf') or np.isnan(vector[2]) or vector[2] == float('inf'):
                    log.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10,63)
                else:
                    if decode_log:
                        sign = -1 if vector[0] > 0.5 else 1
                        try:
                            real_value = math.exp(vector[1]) * sign
                        except OverflowError as e:
                            real_value = pow(10,63) * sign
                    else:
                        real_value = vector[2] * self._abs_mean

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
