import torch
from flair.embeddings import WordEmbeddings
from flair.data import Sentence


def _cat(embed_list):
    return torch.cat(embed_list, dim=0)


def _sum(embed_list):
    return torch.sum(*embed_list)


class AutoEncoder():
    def __init__(self, combine='concat'):
        if combine == 'concat':
            self._combine_fn = _cat
        elif combine == 'sum':
            self._combine_fn = _sum
        else:
            raise ValueError('expected combine to be "concat" or "sum"')

        self._model = WordEmbeddings('glove')

    def encode(self, column_data):
        output = []
        for cell in column_data:
            sent = Sentence(cell)
            self._model.embed(sent)
            vector = self.combine_fn([tok.embedding for tok in sent.tokens])
            output.append(vector)
        return output

    def decoder(self, vectors):
        # TODO
        raise NotImplementedError
