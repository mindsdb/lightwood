import torch
import numpy as np
from scipy.special import softmax
from lightwood.encoder.text.helpers.rnn_helpers import Lang
from lightwood.encoder.base import BaseEncoder

UNCOMMON_WORD = '__mdb_unknown_cat'
UNCOMMON_TOKEN = 0


class OneHotEncoder(BaseEncoder):

    def __init__(self, is_target=False, target_class_distribution=None):
        super().__init__(is_target)
        self._lang = None
        self.rev_map = {}
        if self.is_target:
            self.target_class_distribution = target_class_distribution
            self.index_weights = None

    def prepare(self, priming_data, max_dimensions=20000):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        priming_data = [x if x is not None else UNCOMMON_WORD for x in priming_data]
        self._lang = Lang('default')
        self._lang.index2word = {UNCOMMON_TOKEN: UNCOMMON_WORD}
        self._lang.word2index = {UNCOMMON_WORD: UNCOMMON_TOKEN}
        self._lang.word2count[UNCOMMON_WORD] = 0
        self._lang.n_words = 1
        for category in priming_data:
            if category is not None:
                self._lang.addWord(str(category))

        while self._lang.n_words > max_dimensions:
            necessary_words = UNCOMMON_WORD
            least_occuring_words = self._lang.getLeastOccurring(n=len(necessary_words) + 1)

            word_to_remove = None
            for word in least_occuring_words:
                if word not in necessary_words:
                    word_to_remove = word
                    break

            self._lang.removeWord(word_to_remove)

        if self.is_target:
            self.index_weights = [None] * self._lang.n_words
            if self.target_class_distribution is not None:
                self.index_weights[0] = np.mean(list(self.target_class_distribution.values()))
            else:
                self.index_weights[0] = 1
            for word in set(priming_data):
                if self.target_class_distribution is not None:
                    self.index_weights[self._lang.word2index[str(word)]] = 1 / self.target_class_distribution[word]
                else:
                    self.index_weights[self._lang.word2index[str(word)]] = 1
            self.index_weights = torch.Tensor(self.index_weights)

        self.output_size = self._lang.n_words
        self.rev_map = self._lang.index2word
        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        ret = []
        v_len = self._lang.n_words

        for word in column_data:
            encoded_word = [0] * v_len
            if word is not None:
                word = str(word)
                index = self._lang.word2index[word] if word in self._lang.word2index else UNCOMMON_TOKEN
                encoded_word[index] = 1

            ret.append(encoded_word)

        return torch.Tensor(ret)

    def decode(self, encoded_data, return_raw=False):
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            # Logits and onehots are not the same in definition
            # But this explicitly operates on logits; it will take care of
            # the one hot (so you can pass something in the softmax logit space)
            # But will not affect something that is already OHE.
            ohe_index = np.argmax(vector)
            ret.append(self._lang.index2word[ohe_index])

            if return_raw:
                probs.append(softmax(vector).tolist())

        if return_raw:
            return ret, probs, self.rev_map
        else:
            return ret
