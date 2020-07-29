import math
import logging
import sys

import torch
import numpy as np
from lightwood.encoders.encoder_base import BaseEncoder


class NumericEncoder(BaseEncoder):

    def __init__(self, data_type=None, is_target=False):
        super().__init__(is_target)
        self._type = data_type
        self._abs_mean = None
        self.decode_log = False
        self.extra_outputs = 0

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        value_type = 'int'
        for number in priming_data:
            try:
                number = float(number)
            except:
                continue

            if np.isnan(number):
                err = 'Lightwood does not support working with NaN values !'
                logging.error(err)
                raise Exception(err)

            if int(number) != number:
                value_type = 'float'

        self._type = value_type if self._type is None else self._type
        non_null_priming_data = [float(str(x).replace(',','.')) for x in priming_data if x is not None]
        self._abs_mean = np.mean(np.abs(non_null_priming_data))
        self._prepared = True

    def encode(self, data):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

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
                vector = [0] * (3 + 2 * self.extra_outputs)
                if real is not None and self._abs_mean > 0:
                    vector[0] = 1 if real < 0 else 0
                    vector[1] = math.log(abs(real)) if abs(real) > 0 else - 20
                    vector[2] = real / self._abs_mean
                else:
                    logging.debug(f'Can\'t encode target value: {real}')

            else:
                vector = [0] * 4
                try:
                    if real is None:
                        vector[0] = 0
                    else:
                        vector[0] = 1
                        vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                        vector[2] = 1 if real < 0 else 0
                        vector[3] = real/self._abs_mean
                except Exception as e:
                    vector = [0] * 4
                    logging.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(vector)

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values, decode_log=None):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        ret = []
        if type(encoded_values) != type([]):
            encoded_values = encoded_values.tolist()

        for vector in encoded_values:
            if self.is_target:
                if np.isnan(vector[0]) or vector[0] == float('inf') or np.isnan(vector[1]) or vector[1] == float('inf') or np.isnan(vector[2]) or vector[2] == float('inf'):
                    logging.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10,63)
                else:
                    if decode_log:
                        sign = -1 if vector[0] > 0.5 else 1
                        real_value = [math.exp(vector[i*2 + 1]) * sign for i in range(1 + self.extra_outputs)]
                    else:
                        real_value = [vector[2*i + 2] * self._abs_mean for i in range(1 + self.extra_outputs)]

                    if self._type == 'int':
                        real_value = [int(x) for x in real_value]

                    if len(real_value) < 2:
                        real_value = real_value[0]
            else:
                if vector[0] < 0.5:
                    ret.append(None)
                    continue

                real_value = vector[3] * self._abs_mean

                if self._type == 'int':
                    real_value = round(real_value)

            ret.append(real_value)
        return ret


if __name__ == "__main__":
    data = [1,1.1,2,-8.6,None,0]

    encoder = NumericEncoder()

    encoder.prepare_encoder(data)
    encoded_vals = encoder.encode(data)

    assert(encoded_vals[1][1] > 0)
    assert(encoded_vals[2][1] > 0)
    assert(encoded_vals[3][1] > 0)
    for i in range(0,3):
        assert(encoded_vals[i][2] == 0)
    assert(encoded_vals[3][2] == 1)
    assert(encoded_vals[4][3] == 0)

    decoded_vals = encoder.decode(encoded_vals)

    for i in range(len(encoded_vals)):
        if decoded_vals[i] is None:
            assert(decoded_vals[i] == data[i])
        else:
            np.testing.assert_almost_equal(round(decoded_vals[i],10), round(data[i],10))
