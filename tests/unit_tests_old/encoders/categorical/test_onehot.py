import unittest
from lightwood.encoders.categorical.onehot import (
    OneHotEncoder,
    UNCOMMON_TOKEN,
    UNCOMMON_WORD
)


class TestOnehot(unittest.TestCase):
    def test_encode_and_decode(self):
        data = ['category 1', 'category 3', 'category 4', None]

        enc = OneHotEncoder()

        enc.prepare(data)
        encoded_data = enc.encode(data)
        decoded_data = enc.decode(enc.encode(['category 2', 'category 1', 'category 3', None]))

        self.assertTrue(len(encoded_data) == 4)
        self.assertTrue(decoded_data[1] == 'category 1')
        self.assertTrue(decoded_data[2] == 'category 3')
        for i in [0, 3]:
            self.assertTrue(encoded_data[0][i] == UNCOMMON_TOKEN)
            self.assertTrue(decoded_data[i] == UNCOMMON_WORD)

        # Test max_dimensions
        for max_dimensions in [2, 3]:
            data = ['category 1', 'category 1', 'category 3', 'category 4', 'category 4', 'category 4', None]

            enc = OneHotEncoder()

            enc.prepare(data, max_dimensions=max_dimensions)
            encoded_data = enc.encode(data)
            decoded_data = enc.decode(enc.encode(['category 1', 'category 2', 'category 3', 'category 4', None]))

            if max_dimensions == 3:
                self.assertTrue(decoded_data[0] == 'category 1')
            else:
                self.assertTrue(decoded_data[0] == UNCOMMON_WORD)

            self.assertTrue(decoded_data[1] == UNCOMMON_WORD)
            self.assertTrue(decoded_data[2] == UNCOMMON_WORD)
            self.assertTrue(decoded_data[4] == UNCOMMON_WORD)

            self.assertTrue(decoded_data[3] == 'category 4')
