import numpy as np
from lightwood.encoders import BaseEncoder
from sklearn.preprocessing import MultiLabelBinarizer


class MultihotEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self.max_words_per_sent = None
        self.binarizer = MultiLabelBinarizer()

    def prepare_encoder(self, column_data, max_dimensions=100):
        self.binarizer.fit(column_data)
        self._prepared = True

    def encode(self, column_data):
        data_array = self.binarizer.transform(column_data)
        return self._pytorch_wrapper(data_array)

    def decode(self, probabilities):
        probabilities = np.array(probabilities)
        vectors = np.where(probabilities > 0.5, 1, 0)
        words_tuples = self.binarizer.inverse_transform(vectors)
        return [list(w) for w in words_tuples]
