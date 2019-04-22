import torch

class NumericEncoder:

    def __init__(self, data_type = None):

        self._trained = False
        self._min_value = None
        self._max_value = None
        self._type = data_type
        self._mean = None

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

            vector = [0]*2

            if number is None:
                ret += [vector]
                continue

            vector[-1] = 1 # is not null

            new_number = (number - self._mean)/(self._max_value-self._min_value)
            vector[0] = new_number
            ret += [vector]


        return torch.FloatTensor(ret)


    def decode(self, encoded_values):
        ret = []
        for vector in encoded_values.tolist():
            if vector[-1] == 0: # is not null = false
                ret += [None]
                continue

            encoded_value = vector[0]

            real_value = (self._max_value-self._min_value)*encoded_value + self._mean

            if self._type == 'int':
                real_value = round(real_value)

            ret += [real_value]

        return ret



if __name__ == "__main__":

    encoder = NumericEncoder(data_type='int')

    print(encoder.encode([1,2,2,2,2,2,8.6]))

    print(encoder.decode(encoder.encode([1, 2, 2, 2, 2, 2, 8.7, 800, None])))
