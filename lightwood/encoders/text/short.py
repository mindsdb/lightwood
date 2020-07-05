import torch
from flair.data import Sentence
from lightwood.encoders.categorical import CategoricalAutoEncoder


def _cat(embed_list):
    assert len(embed_list) > 0
    return torch.cat(embed_list, dim=0)


def _sum(embed_list):
    assert len(embed_list) > 0
    return torch.cat([emb[None] for emb in embed_list], dim=0).mean(0)


class TextAutoEncoder(CategoricalAutoEncoder):
    def __init__(self, is_target=False, combine='sum'):
        super().__init__(is_target)

        if combine == 'concat':
            self._combine_fn = _cat
        elif combine == 'sum':
            self._combine_fn = _sum
        else:
            raise ValueError('expected combine to be "concat" or "sum"')
        
    def prepare_encoder(self, column_data):
        no_null_sentences = (x if x is not None else '' for x in column_data)
        unique_words = set()
        for sent in map(Sentence, no_null_sentences):
            for tok in sent.tokens:
                unique_words.add(tok.text)
        super().prepare_encoder(unique_words)

    def encode(self, column_data):
        no_null_sentences = (x if x is not None else '' for x in column_data)
        output = []
        for sent in map(Sentence, no_null_sentences):
            if len(sent) > 0:
                encoded_words = super().encode(list(tok.text for tok in sent.tokens))
            else:
                encoded_words = super().encode([''])
            encoded_sent = self._combine_fn(encoded_words)
            output.append(encoded_sent)
        return output

    def decode(self, vectors):
        raise NotImplementedError
