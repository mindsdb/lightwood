import unittest
import numpy as np
import random
import string

from sklearn.metrics import accuracy_score

from lightwood.encoder.categorical.multihot import MultiHotEncoder


class TestMultiHotEncoder(unittest.TestCase):
    def get_vocab(self):
        return [''.join(random.choices(string.ascii_uppercase, k=5)) for i in range(10)]

    def test_multi_encoding(self):
        vocab = self.get_vocab()
        tags = [list(set(random.choices(vocab, k=random.randint(1, 3)))) for i in range(100)]

        priming_data = tags[:70]
        test_data = tags[70:]

        enc = MultiHotEncoder()
        enc.prepare(priming_data)

        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        test_labels_str = [str(sorted(tags)) for tags in test_data]
        decoded_labels_str = [str(sorted(tags)) for tags in decoded_data]
        encoder_accuracy = accuracy_score(test_labels_str, decoded_labels_str)
        self.assertEqual(encoder_accuracy, 1)

    def test_multi_encoding_empty_row(self):
        vocab = self.get_vocab()
        tags = [list(set(random.choices(vocab, k=random.randint(1, 3)))) for i in range(10)]
        tags.append([])

        enc = MultiHotEncoder()
        enc.prepare(tags)

        encoded_data = enc.encode(tags)
        decoded_data = enc.decode(encoded_data)

        test_labels_str = [str(sorted(t)) for t in tags]
        decoded_labels_str = [str(sorted(t)) for t in decoded_data]
        encoder_accuracy = accuracy_score(test_labels_str, decoded_labels_str)
        self.assertEqual(encoder_accuracy, 1)

    def test_handle_unseen_none(self):
        vocab = self.get_vocab()
        tags = [list(set(random.choices(vocab, k=random.randint(1, 3)))) for i in range(10)]
        tags.append(None)

        train_tags = tags[:-1]
        test_tags = tags[-1:]

        enc = MultiHotEncoder()
        enc.prepare(train_tags)

        encoded_data = enc.encode(test_tags)
        decoded_data = enc.decode(encoded_data)
        assert (np.array(encoded_data[0]) == 0).all()
        assert decoded_data[0] == []
