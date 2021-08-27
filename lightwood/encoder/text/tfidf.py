import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from lightwood.encoder.base import BaseEncoder


class TfidfEncoder(BaseEncoder):
    def __init__(self, is_target: bool = False):
        super().__init__(is_target)
        self.ngram_range = (1, 5)
        self.max_features = 500

    def prepare(self, priming_data, training_data=None):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self.tfidf_vectorizer.fit_transform([str(x) for x in priming_data])

    def encode(self, column_data):
        transformed_data = self.tfidf_vectorizer.transform([str(x) for x in column_data])
        dense_transformed_data = [np.array(x.todense())[0] for x in transformed_data]
        return torch.Tensor(dense_transformed_data)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')
