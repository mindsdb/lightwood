import torch
import math
import logging
import sys


class NumericEncoder:

    def __init__(self, data_type=None, is_target=False):
        self._type = data_type
        self._min_value = None
        self._max_value = None
        self._mean = None
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        count = 0
        value_type = 'int'
        for number in priming_data:
            try:
                number = float(number)
            except:
                continue

            if math.isnan(number):
                err = 'Lightwood does not support working with NaN values !'
                logging.error(err)
                raise Exception(err)

            self._min_value = number if self._min_value is None or self._min_value > number else self._min_value
            self._max_value = number if self._max_value is None or self._max_value < number else self._max_value
            count += number

            if int(number) != number:
                value_type = 'float'

        self._type = value_type if self._type is None else self._type
        self._mean = count / len(priming_data)
        self._prepared = True

    def encode(self, data):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')
        ret = []

        for number in data:
            vector_len = 4
            vector = [0]*vector_len

            if number is None:
                vector[3] = 0
                ret.append(vector)
                continue
            else:
                vector[3] = 1

            try:
                number = float(number)
            except:
                logging.warning('It is assuming that  "{what}" is a number but cannot cast to float'.format(what=number))
                ret.append(vector)
                continue

            if number < 0:
                vector[0] = 1

            if number == 0:
                vector[2] = 1
            else:
                vector[1] = math.log(abs(number))

            ret.append(vector)

        return self._pytorch_wrapper(ret)


    def decode(self, encoded_values):
        ret = []
        for vector in encoded_values.tolist():
            if not math.isnan(vector[0]):
                is_negative = True if abs(round(vector[0])) == 1 else False
            else:
                logging.warning(f'Occurance of `nan` value in encoded numerical value: {vector}')
                is_negative = False

            if not math.isnan(vector[1]):
                encoded_nr = vector[1]
            else:
                logging.warning(f'Occurance of `nan` value in encoded numerical value: {vector}')
                encoded_nr = 0

            if not math.isnan(vector[2]):
                is_zero = True if abs(round(vector[2])) == 1 else False
            else:
                logging.warning(f'Occurance of `nan` value in encoded numerical value: {vector}')
                is_zero = False

            if not math.isnan(vector[3]):
                is_none = True if abs(round(vector[3])) == 0 else False
            else:
                logging.warning(f'Occurance of `nan` value in encoded numerical value: {vector}')
                is_none = True

            if is_none:
                ret.append(None)
                continue

            if is_zero:
                ret.append(0)
                continue

            try:
                real_value = math.exp(encoded_nr)
                if is_negative:
                    real_value = -real_value
            except:
                if self._type == 'int':
                    real_value = pow(2,63)
                else:
                    real_value = float('inf')

            if self._type == 'int':
                real_value = round(real_value)

            ret.append(real_value)

        return ret



if __name__ == "__main__":
    data = [1,1.1,2,-8.6,None,0]

    encoder = NumericEncoder()

    encoder.prepare_encoder(data)
    encoded_vals = encoder.encode(data)

    assert(sum(encoded_vals[4]) == 0)
    assert(sum(encoded_vals[0]) == 1)
    assert(encoded_vals[1][1] > 0)
    assert(encoded_vals[2][1] > 0)
    assert(encoded_vals[3][1] > 0)
    for i in range(0,4):
        assert(encoded_vals[i][3] == 1)
    assert(encoded_vals[4][3] == 0)

    decoded_vals = encoder.decode(encoded_vals)

    for i in range(len(encoded_vals)):
        if decoded_vals[i] is None:
            assert(decoded_vals[i] == data[i])
        else:
            assert(round(decoded_vals[i],5) == round(data[i],5))
