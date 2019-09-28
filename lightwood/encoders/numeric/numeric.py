import torch
import math
import logging

class NumericEncoder:

    def __init__(self, data_type=None):
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
                logging.error('Lightwood does not support working with NaN values !')
                exit()

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
            if vector[3] == 1:
                ret.append(None)
                continue

            if math.isnan(vector[1]):
                abs_rounded_first = 0
            else:
                abs_rounded_first = abs(round(vector[1]))

            if abs_rounded_first == 1:
                real_value = 0
            else:
                if math.isnan(vector[0]):
                    abs_rounded_zero = 0
                else:
                    abs_rounded_zero = abs(round(vector[0]))

                is_negative = True if abs_rounded_zero == 1 else False
                encoded_value = vector[2]
                try:
                    real_value = -math.exp(encoded_value) if is_negative else math.exp(encoded_value)
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
    data = [1,1.1,2,8.6,None]

    encoder = NumericEncoder()

    encoder.fit(data)
    encoded_vals = encoder.encode(data)

    assert(sum(encoded_vals[0]) == 0)
    assert(encoded_vals[1][2] > 0)
    assert(encoded_vals[2][2] > 0)
    assert(encoded_vals[3][2] > 0)
    for i in range(0,4):
        assert(encoded_vals[i][3] == 0)
    assert(encoded_vals[4][3] == 1)

    decoded_vals = encoder.decode(encoded_vals)
    for i in range(len(encoded_vals)):
        if decoded_vals[i] is None:
            assert(decoded_vals[i] == data[i])
        else:
            assert(round(decoded_vals[i],5) == round(data[i],5))
