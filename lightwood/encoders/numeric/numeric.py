import torch
import math

class NumericEncoder:

    def __init__(self, data_type = None, is_target = False):
        self._is_target = True
        self._trained = False
        self._min_value = None
        self._max_value = None
        self._type = data_type
        self._mean = None
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, data):

        if self._trained == False:
            count = 0
            value_type = 'int'
            for number in data:
                if number is None:
                    continue
                self._min_value = number if self._min_value is None or self._min_value > number else self._min_value
                self._max_value = number if self._max_value is None or self._max_value < number else self._max_value
                count += number

                if int(number) != number:
                    value_type = 'float'

            self._type = value_type if self._type is None else self._type
            self._mean = count / len(data)

            self._trained = True

        ret = []

        for number in data:

            vector = [0]*4

            if number is None:
                ret += [vector]
                continue

            if number < 0:
                vector[0] = 1

            if number == 0:
                vector[1] = 1

            else:
                vector[2] = math.log(abs(number))

            vector[-1] = 1 # is not null



            if self._is_target:
                vector=vector[:-1]

            ret += [vector]


        return self._pytorch_wrapper(ret)


    def decode(self, encoded_values):
        ret = []
        for vector in encoded_values.tolist():
            if vector[-1] == 0 and self._is_target == False: # is not null = false
                ret += [None]
                continue


            if abs(round(vector[1])) == 1:
                real_value = 0
            else:
                is_negative = True if abs(round(vector[0])) == 1 else False
                encoded_value = vector[2]
                real_value = -math.exp(encoded_value) if is_negative else math.exp(encoded_value) #(self._max_value-self._min_value)*encoded_value + self._mean


            if self._type == 'int':
                real_value = round(real_value)

            ret += [real_value]

        return ret



if __name__ == "__main__":

    encoder = NumericEncoder(data_type='int')

    print(encoder.encode([1,2,2,2,2,2,8.6]))

    print(encoder.decode(encoder.encode([1, 2, 2, 2, 2, 2, 8.7, 800, None])))

    encoder = NumericEncoder()

    print(encoder.encode([1, 2, 2, 2, 2, 2, 8.6]))

    print(encoder.decode(encoder.encode([1, 2, 2, 2, 2, 2, 8.7, None])))
