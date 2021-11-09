import unittest
import torch
from torch import Tensor
import numpy as np
from lightwood.encoder.categorical.binary import (
    BinaryEncoder,
)

from lightwood.helpers.constants import _UNCOMMON_WORD, _UNCOMMON_TOKEN

class TestBinary(unittest.TestCase):
    """ Test the OHE vector encoder """

    def test_encode_decode_with_binary(self):
        """

        """    
        # Generate random list of T/F
        data = ["apple", "apple", "orange", None, "apple", "orange"]
        test_data = ["apple", "banana", "orange", None, "apple"]
        ytest = ["apple", _UNCOMMON_WORD, "orange", _UNCOMMON_WORD, "apple"]

        enc = BinaryEncoder()
        enc.prepare(data)

        enc_data = enc.encode(data)
        dec_data = enc.decode(enc.encode(test_data))

        self.assertTrue((enc_data == Tensor([
            [1., 0.,],  # category 1
            [1., 0.,],  # category 3
            [0., 1.,],  # category 4
            [0., 0.,],  # None
            [1., 0.,],  # category 3
            [0., 1.,],  # category 3
        ])).all())

        for i in range(len(ytest)):
            self.assertTrue(dec_data[i] == ytest[i])

    def check_only_binary(self):
        """ Ensure binary strictly enforces binary typing """
        data = ["apple", "apple", "orange", "banana", "apple", "orange"]

        enc = BinaryEncoder()
        self.assertRaises(ValueError, enc.prepare(data))

    def check_probabilities(self):
        """
        """
        # Generate random list of T/F
        np.random.seed(1)
        data = np.random.rand(10) > 0.5

        enc = BinaryEncoder()
        enc.prepare(data)

        # Make data to represent random weights that do not sum to 1
        torch.manual_seed(1)
        wt_vec = torch.rand(size=(len(data), len(enc.map)))*10

        _, probs, _ = enc.decode_probabilities(wt_vec)
        self.assertTrue(np.all([np.isclose(sum(i), 1) for i in probs]))

if __name__ == "__main__":
    unittest.main()
