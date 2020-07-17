import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from lightwood.constants.lightwood import ENCODER_AIM
from lightwood.encoders.encoder_base import BaseEncoder


class TfidfEncoder(BaseEncoder):
    def __init__(self, is_target=False, aim=ENCODER_AIM.BALANCE):
        super().__init__(is_target)
        self.aim = aim
        self._pytorch_wrapper = torch.FloatTensor
        if self.aim == ENCODER_AIM.SPEED:
            self.ngram_range = (1,3)
            self.max_features = 200
        elif self.aim == ENCODER_AIM.BALANCE:
            self.ngram_range = (1,5)
            self.max_features = 500
        elif self.aim == ENCODER_AIM.ACCURACY:
            self.ngram_range = (1,8)
            self.max_features = None

    def prepare_encoder(self, priming_data, training_data=None):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self.tfidf_vectorizer.fit_transform([str(x) for x in priming_data])

    def encode(self, column_data):
        transformed_data = self.tfidf_vectorizer.transform([str(x) for x in column_data])
        dense_transformed_data = [np.array(x.todense())[0] for x in transformed_data]
        return self._pytorch_wrapper(dense_transformed_data)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')


if __name__ == "__main__":
    import random
    import string

    random.seed(2)
    text = [''.join(random.choices(string.printable, k=random.randint(5,500))) for x in range(1000)]

    enc = TfidfEncoder()
    enc.prepare_encoder(text)
    encoded_data = enc.encode(text)
    print(encoded_data)
