from typing import List
import torch
from lightwood.encoder import BaseEncoder
from lightwood.encoder.categorical import CategoricalAutoEncoder
from lightwood.helpers.text import tokenize_text
from lightwood.helpers.torch import concat_vectors_and_pad, average_vectors
import pandas as pd


class ShortTextEncoder(BaseEncoder):
    def __init__(self, is_target=False, mode=None):
        """
        :param is_target:
        :param mode:
            None or "concat" or "mean".
            When None, it will be set automatically based on is_target:
            (is_target) -> 'concat'
            (not is_target) -> 'mean'
        """
        super().__init__(is_target)

        if mode is None:
            if is_target:
                self._mode = 'concat'
            else:
                self._mode = 'mean'
        else:
            if mode not in ['concat', 'mean']:
                self._unexpected_mode()

            if is_target and mode != 'concat':
                raise ValueError('mode must be "concat" when is_target=True')

            self._mode = mode

        # Defined in self.prepare()
        self._combine_fn = None
        self.max_words_per_sent = None
        self.cae = CategoricalAutoEncoder(is_target=is_target, max_encoded_length=100)
        self.is_prepared = False

    def _unexpected_mode(self):
        raise ValueError('unexpected combine value (must be "mean" or "concat")')

    # defining both of these as normal functions because pickle can't deal with lambdas
    def _combine_concat(self, vecs):
        return concat_vectors_and_pad(vecs, self.max_words_per_sent)

    def _combine_mean(self, vecs):
        return average_vectors(vecs)

    def prepare(self, priming_data):
        no_null_sentences = (x if x is not None else '' for x in priming_data)
        unique_tokens = set()
        max_words_per_sent = 0
        for sent in no_null_sentences:
            tokens = tokenize_text(sent)
            max_words_per_sent = max(max_words_per_sent, len(tokens))
            for tok in tokens:
                unique_tokens.add(tok)

        self.cae.prepare(pd.Series(list(unique_tokens)), pd.Series([]))

        if self._mode == 'concat':
            self.max_words_per_sent = max_words_per_sent
            self._combine_fn = self._combine_concat
        elif self._mode == 'mean':
            self._combine_fn = self._combine_mean
        else:
            self._unexpected_mode()

        self.is_prepared = True
        encoded = self.encode([priming_data[0]])
        self.output_size = len(encoded[0])

    def encode(self, column_data: List[str]) -> torch.Tensor:
        no_null_sentences = (x if x is not None else '' for x in column_data)
        output = []
        for sent in no_null_sentences:
            tokens = tokenize_text(sent)
            encoded_words = self.cae.encode(tokens)
            encoded_sent = self._combine_fn(encoded_words)
            output.append(torch.Tensor(encoded_sent))
        output = torch.stack(output)
        return output

    def decode(self, vectors):
        if self._mode == 'concat':

            vec_size = self.cae.output_size

            output = []
            for vec in vectors:

                viewed_vec = vec.view(-1, vec_size)

                # Find index of first padding vector
                for index, v in enumerate(viewed_vec):
                    if v.abs().sum() == 0:
                        break
                else:
                    index = viewed_vec.size(0)

                out = self.cae.decode(
                    viewed_vec[:index]
                )

                output.append(out)

            return output

        elif self._mode == 'mean':
            raise ValueError('decode is only defined for mode="concat"')
        else:
            self._unexpected_mode()
