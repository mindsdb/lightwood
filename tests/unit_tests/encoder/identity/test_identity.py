import unittest
import numpy as np
from torch import Tensor
from typing import List
from lightwood.encoder.numeric import IdentityEncoder


class TestIdentityEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        data = [1, 1.1, 2, -8.6, 0]

        encoder = IdentityEncoder()

        encoded_vals = encoder.encode(data)

        self.assertTrue(isinstance(encoded_vals, Tensor))

        for i in range(0, 5):
            np.testing.assert_almost_equal(round(encoded_vals[i], 10), round(data[i], 10))

        decoded_vals = encoder.decode(encoded_vals)

        self.assertTrue(isinstance(decoded_vals, List[object]))

        for i in range(len(encoded_vals)):
            np.testing.assert_almost_equal(round(decoded_vals[i], 10), round(data[i], 10))
