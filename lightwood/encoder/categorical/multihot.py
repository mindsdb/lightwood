import torch
import numpy as np
from lightwood.encoders import BaseEncoder
from sklearn.preprocessing import MultiLabelBinarizer


class MultiHotEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._binarizer = MultiLabelBinarizer()
        self._seen = set()

    @staticmethod
    def _clean_col_data(column_data):
        column_data = [ (arr if arr is not None else []) for arr in column_data]
        column_data = [ [str(x) for x in arr] for arr in column_data]
        return column_data

    def prepare(self, column_data, max_dimensions=100):
        column_data = self._clean_col_data(column_data)
        self._binarizer.fit(column_data + [('None')])
        for arr in column_data:
            for x in arr:
                self._seen.add(x)
        self._prepared = True

    def encode(self, column_data):
        column_data = self._clean_col_data(column_data)
        data_array = self._binarizer.transform(column_data)
        return torch.Tensor(data_array)

    def decode(self, vectors):
        # It these are logits output by the neural network, we need to treshold them to binary vectors
        vectors = np.where(vectors > 0, 1, 0)
        words_tuples = self._binarizer.inverse_transform(vectors)
        return [list(w) for w in words_tuples]
