import torch
import numpy as np
from lightwood.encoders import BaseEncoder
from sklearn.preprocessing import MultiLabelBinarizer


class MultiHotEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._binarizer = MultiLabelBinarizer()

    def prepare_encoder(self, column_data, max_dimensions=100):
        self._binarizer.fit(column_data)
        self._prepared = True

    def encode(self, column_data):
        column_data = [ (c if c is not None else []) for c in column_data]
        data_array = self._binarizer.transform(column_data)
        return self._pytorch_wrapper(data_array)

    def decode(self, vectors):
        # It these are logits output by the neural network, we need to treshold them to binary vectors
        vectors = np.where(vectors > 0, 1, 0)
        words_tuples = self._binarizer.inverse_transform(vectors)
        return [list(w) for w in words_tuples]
