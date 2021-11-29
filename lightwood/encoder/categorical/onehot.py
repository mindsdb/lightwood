import torch
import numpy as np
from scipy.special import softmax
from lightwood.encoder.text.helpers.rnn_helpers import Lang
from lightwood.helpers.log import log
from lightwood.encoder.base import BaseEncoder

UNCOMMON_WORD = '__mdb_unknown_cat'
UNCOMMON_TOKEN = 0


class OneHotEncoder(BaseEncoder):
    """

    Why are we handling target weighting inside encoders? Simple: we'd otherwise have to compute per-index weighting inside the mixers, rather than having that code unified inside 2x encoders. So moving this to the mixer will still involve having to pass the target encoder to the mixer, but will add the additional complexity of having to pass a weighting map to the mixer and adding class-to-index translation boilerplate + weight setting for each mixer
    """ # noqa
    def __init__(self, is_target=False, target_weights=None, handle_unknown='unknown_token'):
        super().__init__(is_target)
        self._lang = None
        self.rev_map = {}

        if handle_unknown not in {"unknown_token", "return_zeros"}:
            raise ValueError(f"handle_unknown should be either 'unknown_token' or 'return_zeros', got {handle_unknown}")
        else:
            self.handle_unknown = handle_unknown

        if self.is_target:
            if self.handle_unknown != 'unknown_token':
                raise ValueError(f'One Hot Encoders used for target encoding can only be used with `handle_unknown` \
                                   set to `unknown_token`. The option: "{self.handle_unknown}" is not supported!')

        self.target_weights = None
        self.index_weights = None
        if self.is_target:
            self.target_weights = target_weights

    def prepare(self, priming_data, max_dimensions=20000):
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self._lang = Lang('default')
        if self.handle_unknown == "return_zeros":
            priming_data = [x for x in priming_data if x is not None]
            self._lang.index2word = {}
            self._lang.word2index = {}
            self._lang.n_words = 0
        elif self.handle_unknown == "unknown_token":
            priming_data = [x if x is not None else UNCOMMON_WORD for x in priming_data]
            self._lang.index2word = {UNCOMMON_TOKEN: UNCOMMON_WORD}
            self._lang.word2index = {UNCOMMON_WORD: UNCOMMON_TOKEN}
            self._lang.word2count[UNCOMMON_WORD] = 0
            self._lang.n_words = 1

        for category in priming_data:
            if category is not None:
                self._lang.addWord(str(category))

        while self._lang.n_words > max_dimensions:
            if self.handle_unknown == "return_zeros":
                necessary_words = []
            elif self.handle_unknown == "unknown_token":
                necessary_words = [UNCOMMON_WORD]
            least_occuring_words = self._lang.getLeastOccurring(n=len(necessary_words) + 1)

            word_to_remove = None
            for word in least_occuring_words:
                if word not in necessary_words:
                    word_to_remove = word
                    break

            self._lang.removeWord(word_to_remove)

        # Note: Is target assume that we are operating in "unknown_token" mode
        if self.is_target:
            self.index_weights = [1] * self._lang.n_words
            if self.target_weights is not None:
                uncommon_weight = np.min(list(self.target_weights.values()))
                self.index_weights[0] = uncommon_weight
                self.target_weights[UNCOMMON_WORD] = uncommon_weight
            for word in set(priming_data):
                if self.target_weights is not None:
                    self.index_weights[self._lang.word2index[str(word)]] = \
                        self.target_weights[word] / np.max(self.target_weights.values())

            self.index_weights = torch.Tensor(self.index_weights)

        self.output_size = self._lang.n_words
        self.rev_map = self._lang.index2word
        self.is_prepared = True

    def encode(self, column_data):
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        v_len = self._lang.n_words

        for word in column_data:
            encoded_word = [0] * v_len
            if word is not None:
                word = str(word)
                if self.handle_unknown == "return_zeros":
                    if word in self._lang.word2index:
                        index = self._lang.word2index[word]
                        encoded_word[index] = 1
                    else:
                        # Encoding an unknown value will result in a vector of zeros
                        log.warning('Trying to encode a value never seen before, returning vector of zeros')
                else:  # self.handle_unknown == "unknown_token"
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

            all_zeros = not np.any(vector)
            if self.handle_unknown == "return_zeros" and all_zeros:
                ret.append(UNCOMMON_WORD)
            else:  # self.handle_unknown == "unknown_token"
                ohe_index = np.argmax(vector)
                ret.append(self._lang.index2word[ohe_index])

            if return_raw:
                probs.append(softmax(vector).tolist())

        if return_raw:
            return ret, probs, self.rev_map
        else:
            return ret
