import torch
from lightwood.encoders import BaseEncoder

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
import flair


class FlairEmbeddingEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self.max_sentence = 768

    def prepare_encoder(self, column_data):
        # @TODO Maybe allow for an `aim` parameter and set a simpler embedding if the aim is speed (see img_2_vec as an example)
        self.embedding = TransformerDocumentEmbeddings('roberta-base')

    def encode(self, column_data):
        vec = []
        for data in column_data:
            if data is not None:
                data = data[0:self.max_sentence]
            sentence = Sentence(data)
            with torch.no_grad():
                self.embedding.embed(sentence)
                vec.append(sentence.get_embedding().to('cpu').clone())
                sentence.clear_embeddings()
        return torch.stack(vec)

    def decode(self, vectors):
        raise Exception('This encoder is not bi-directional')
