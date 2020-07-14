from lightwood.encoders import BaseEncoder, CategoricalAutoEncoder
from lightwood.encoders.text import ShortTextEncoder
from lightwood.helpers.torch import concat_vectors_and_pad


class TagsEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self.cae = CategoricalAutoEncoder(is_target, max_encoded_length=100)

        self.max_words_per_sent = None

    def combine_vectors(self, vectors):
        return concat_vectors_and_pad(vectors, self.max_words_per_sent)

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
        self.max_words_per_sent = max_words_per_sent

    def encode(self, column_data):
        no_null_sen = (x if x else [] for x in column_data)
        output = []
        for tags in no_null_sen:
            encoded_words = self.cae.encode(tags)
            encoded_sent = self.combine_vectors(encoded_words)
            output.append(encoded_sent)
        return output

    def decode(self, vectors):

        if self.cae.use_autoencoder:
            vec_size = self.cae.max_encoded_length
        else:
            vec_size = len(self.cae.onehot_encoder._lang.index2word)

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

if __name__ == "__main__":
    # Generate some tests data
    import logging
    import random
    import string
    from sklearn.metrics import accuracy_score

    logging.getLogger().setLevel(logging.DEBUG)

    random.seed(2)

    vocab = [''.join(random.choices(string.ascii_uppercase, k=5)) for i in range(10)]
    tags = [random.choices(vocab, k=random.randint(1, 3)) for i in range(100)]

    priming_data = tags[:70]
    test_data = tags[70:]

    random.shuffle(priming_data)
    random.shuffle(test_data)

    enc = TagsEncoder()

    enc.prepare_encoder(priming_data)
    encoded_data = enc.encode(test_data)
    decoded_data = enc.decode(encoded_data)

    encoder_accuracy = accuracy_score(list(map(str,test_data)), list(map(str,decoded_data)))
    print(f'Categorical encoder accuracy for: {encoder_accuracy} on testing dataset')
    assert(encoder_accuracy > 0.80)
