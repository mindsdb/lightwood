import unittest
import torch
from torch import Tensor
import numpy as np
from lightwood.encoder.categorical.binary import (
    BinaryEncoder,
)

from lightwood.helpers.constants import _UNCOMMON_WORD


class TestBinary(unittest.TestCase):
    """ Test the OHE vector encoder """

    def test_encode_decode_with_binary(self):
        """
        Test binary example case.
        """
        # Generate categories with None
        data = ["apple", "apple", "orange", None, "apple", "orange"]

        # Generate test category with unseen examples (banana, None)
        test_data = ["apple", "banana", "orange", None, "apple"]

        # Ground-truth answer
        ytest = ["apple", _UNCOMMON_WORD, "orange", _UNCOMMON_WORD, "apple"]

        enc = BinaryEncoder()
        enc.prepare(data)

        enc_data = enc.encode(data)
        dec_data = enc.decode(enc.encode(test_data))

        self.assertTrue(
            (
                enc_data
                == Tensor(
                    [
                        [
                            1.0,
                            0.0,
                        ],  # category 1
                        [
                            1.0,
                            0.0,
                        ],  # category 1
                        [
                            0.0,
                            1.0,
                        ],  # category 2
                        [
                            0.0,
                            0.0,
                        ],  # None
                        [
                            1.0,
                            0.0,
                        ],  # category 1
                        [
                            0.0,
                            1.0,
                        ],  # category 2
                    ]
                )
            ).all()
        )

        for i in range(len(ytest)):
            self.assertTrue(dec_data[i] == ytest[i])

    def test_check_only_binary(self):
        """ Ensure binary strictly enforces binary typing """
        data = ["apple", "apple", "orange", "banana", "apple", "orange"]

        enc = BinaryEncoder()
        self.assertRaises(ValueError, enc.prepare, data)

    def test_check_probabilities(self):
        """
        Check whether decode_probabilities returns valid scaled-to-1 terms
        """
        # Generate random list of T/F
        np.random.seed(1)
        data = np.random.rand(10) > 0.5

        enc = BinaryEncoder()
        enc.prepare(data)

        # Make data to represent random weights that do not sum to 1
        torch.manual_seed(1)
        wt_vec = torch.rand(size=(len(data), len(enc.map))) * 10

        _, probs, _ = enc.decode_probabilities(wt_vec)
        self.assertTrue(np.all([np.isclose(sum(i), 1) for i in probs]))

    def test_target_distro_scaled_to_1(self):
        """
        Check whether target distribution passed and handled properly
        """
        data = ["apple", "apple", "orange", "apple", "apple", "orange"]

        # Scaled weights (sum to 1)
        tweights = {"apple": 4 / 6, "orange": 2 / 6}

        enc = BinaryEncoder(is_target=True, target_weights=tweights)
        enc.prepare(data)

        # Get the ground-truth weights
        iweights = torch.ones(size=(len(tweights),))
        for key, value in tweights.items():  # accounts for order
            iweights[enc.map[key]] = value

        # Check inverse weights correct
        self.assertTrue(np.all(((enc.index_weights - iweights) == 0).tolist()))

    def test_distro_nonzeroweights(self):
        """
        Tests if target weights do not sum to 1 properly handled.

        This handles cases where people may choose 1/class_size for weights
        """
        data = ["apple", "apple", "orange", "apple", "apple", "orange"]
        tweights = {"apple": 100, "orange": 5000}

        enc = BinaryEncoder(is_target=True, target_weights=tweights)
        enc.prepare(data)

        # Get the ground-truth weights
        iweights = torch.ones(size=(len(tweights),))
        for key, value in tweights.items():  # accounts for order
            iweights[enc.map[key]] = value

        self.assertTrue(np.all(((enc.index_weights - iweights) == 0).tolist()))

    def test_distro_zero(self):
        """
        Tests edge cause where target weights have a 0 weight which is unacceptable for downstream processing (inverse weights will 1/0)
        """ # noqa
        data = ["apple", "apple", "orange", "apple", "apple", "orange"]

        # Arbitrary weights (ex: number of examples)
        tweights = {"apple": 100, "orange": 0}

        enc = BinaryEncoder(is_target=True, target_weights=tweights)

        # Check if 0-weight class is rejected
        self.assertRaises(ValueError, enc.prepare, data)


if __name__ == "__main__":
    unittest.main()
