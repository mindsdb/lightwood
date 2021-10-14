import unittest
import torch
from typing import List
from lightwood.encoder.identity.identity import IdentityEncoder


class TestIdentityEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        data = [1, 1.1, 2, -8.6, 0]

        encoder = IdentityEncoder()

        encoded_vals = encoder.encode(data)

        self.assertTrue(isinstance(encoded_vals, torch.Tensor))

        for i in range(len(encoded_vals)):
            self.assertTrue(abs(encoded_vals[i].item() - data[i]) <= 1e-5)

        decoded_vals = encoder.decode(encoded_vals)

        self.assertTrue(isinstance(decoded_vals, List))

        for i in range(len(decoded_vals)):
            self.assertTrue(abs(decoded_vals[i] - data[i]) <= 1e-5)
