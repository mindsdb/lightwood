import torch

from lightwood.encoders import BaseEncoder, CategoricalAutoEncoder
from lightwood.helpers.torch import concat_vectors_and_pad


class TagsEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self.cae = CategoricalAutoEncoder(is_target, max_encoded_length=100)

        if combine not in ['mean', 'concat']:
            self._unexpected_combine()

        self._combine = combine

        # Defined in self.prepare_encoder()
        self._combine_fn = None

    def prepare_encoder(self, column_data):
        no_null_sentences = (x if x else [] for x in column_data)
        unique_tokens = set()
        max_words_per_sent = 0
        for tags in no_null_sentences:
            if len(tags) > max_words_per_sent:
                max_words_per_sent = len(tags)
            for tok in tags:
                unique_tokens.add(tok)

        self.cae.prepare_encoder(unique_tokens)
        self._combine_fn = lambda vecs: concat_vectors_and_pad(vecs, max_words_per_sent)

    def encode(self, column_data):
        no_null_sen = (x if x else [] for x in column_data)
        output = []
        for tags in no_null_sen:
            encoded_words = self.cae.encode(tags)
            encoded_sent = self._combine_fn(encoded_words)
            output.append(encoded_sent)
        return output

    def decode(self, vectors):
        output = super().decode(vectors)
        return output
