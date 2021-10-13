import torch
import numpy as np
from scipy.special import softmax
from lightwood.encoder.base import BaseEncoder


# Exists mainly for datasets with loads of binary flags where OHE can be too slow to fit
class BinaryEncoder(BaseEncoder):

    def __init__(self, is_target=False, target_class_distribution=None):
        super().__init__(is_target)
        self.map = {}
        self.rev_map = {}
        self.output_size = 2
        if self.is_target:
            self.target_class_distribution = target_class_distribution
            self.index_weights = None

    def prepare(self, priming_data):
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        for x in priming_data:
            x = str(x)
            if x not in self.map:
                self.map[x] = len(self.map)
                self.rev_map[len(self.rev_map)] = x

            if len(self.map) == 2:
                break

        if self.is_target:
            self.index_weights = [None, None]
            for word in self.map:
                if self.target_class_distribution is not None:
                    self.index_weights[self.map[word]] = 1 / self.target_class_distribution[word]
                else:
                    self.index_weights[self.map[word]] = 1

            self.index_weights = torch.Tensor(self.index_weights)

        self.is_prepared = True

    def encode(self, column_data):
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        ret = []

        for word in column_data:
            index = self.map.get(word, None)
            ret.append([0, 0])
            if index is not None:
                ret[-1][index] = 1

        return torch.Tensor(ret)

    def decode(self, encoded_data, return_raw=False):
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            ret.append(self.rev_map[np.argmax(vector)])

            if return_raw:
                probs.append(softmax(vector).tolist())

        if return_raw:
            return ret, probs, self.rev_map
        else:
            return ret
