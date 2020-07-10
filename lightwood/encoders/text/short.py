import torch
from torch.nn.functional import pad
from lightwood.encoders.categorical import CategoricalAutoEncoder


def _get_tokens(text):
    if len(text) > 0:
        return text.split(' ')
    else:
        return ['']


def _concat(vec_list, max_):
    assert len(vec_list) > 0
    assert len(vec_list) <= max_
    assert max_ > 0

    cat_vec = torch.cat(list(vec_list), dim=0)

    pad_size = max_ - len(vec_list)
    padding = (0, pad_size * vec_list[0].size(0))
    padded = pad(cat_vec[None], padding, 'constant', 0)[0]

    return padded


def _mean(vec_list):
    assert len(vec_list) > 0
    return torch.cat([emb[None] for emb in vec_list], dim=0).mean(0)


class TextAutoEncoder(CategoricalAutoEncoder):
    def __init__(self, is_target=False, combine='mean'):
        super().__init__(is_target)

        if combine not in ['mean', 'concat']:
            self._unexpected_combine()
        
        self._combine = combine

        # Defined in self.prepare_encoder()
        self._combine_fn = None
    
    def _unexpected_combine(self):
        raise ValueError('unexpected combine value (must be "mean" or "concat")')
        
    def prepare_encoder(self, column_data):
        no_null_sentences = (x if x is not None else '' for x in column_data)
        unique_tokens = set()
        max_words_per_sent = 0
        for sent in no_null_sentences:
            tokens = _get_tokens(sent)
            if len(tokens) > max_words_per_sent:
                max_words_per_sent = len(tokens)
            for tok in tokens:
                unique_tokens.add(tok)

        super().prepare_encoder(unique_tokens)

        if self._combine == 'concat':
            self._combine_fn = lambda vecs: _concat(vecs, max_words_per_sent)
        elif self._combine == 'mean':
            self._combine_fn = lambda vecs: _mean(vecs)
        else:
            self._unexpected_combine()

    def encode(self, column_data):
        no_null_sentences = (x if x is not None else '' for x in column_data)
        output = []
        for sent in no_null_sentences:
            tokens = _get_tokens(sent)
            with torch.no_grad():
                encoded_words = super().encode(tokens)
                encoded_sent = self._combine_fn(encoded_words)
            output.append(encoded_sent)
        return output

    def decode(self, vectors):
        if self._combine == 'concat':

            if self.use_autoencoder:
                vec_size = self.max_encoded_length
            else:
                vec_size = len(self.onehot_encoder._lang.index2word)

            output = []
            for vec in vectors:

                viewed_vec = vec.view(-1, vec_size)

                # Find index of first padding vector
                for index, v in enumerate(viewed_vec):
                    if v.abs().sum() == 0:
                        break
                else:
                    index = viewed_vec.size(0)

                out = super().decode(
                    viewed_vec[:index]
                )

                output.append(out)

            return output

        elif self._combine == 'mean':
            raise ValueError('decode is only defined for combine="concat"')
        else:
            self._unexpected_combine()
