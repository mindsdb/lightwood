import torch
import numpy as np
from scipy.special import softmax
from lightwood.encoder.base import BaseEncoder


# Exists mainly for datasets with loads of binary flags where OHE can be too slow to fit
class BinaryEncoder(BaseEncoder):

    def __init__(self, is_target=False, target_class_distribution=None):
        super().__init__(is_target)
        self.map = {}
        self.predict_proba = False
        self.rev_map = {}
        if self.is_target:
            self.target_class_distribution = target_class_distribution
            self.index_weights = None

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')
        
        for x in priming_data:
            i = 0
            x = str(x)
            if x not in self.map:
                self.map[x] = i
                self.rev_map[i] = x
                i += 1
            if len(self.map) > 1:
                break

        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        ret = []

        for word in column_data:
            index = self.map.get(word, None)
            ret.append([0,0])
            if index is not None:
                ret[-1][index] = 1  

        return torch.Tensor(ret)

    def decode(self, encoded_data):
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            ret.append(self.rev_map[np.argmax(vector)])

            if self.predict_proba:
                probs.append(softmax(vector).tolist())

        if self.predict_proba:
            return ret, probs, self.rev_map
        else:
            return ret
