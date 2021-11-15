import unittest
import torch
from torch import Tensor
import numpy as np
from lightwood.encoder.categorical.onehot import (
    OneHotEncoder,
)

from lightwood.helpers.constants import _UNCOMMON_WORD, _UNCOMMON_TOKEN


class TestOnehot(unittest.TestCase):
    """ Test the OHE vector encoder """

    def test_encode_and_decode_with_unknown_token(self):
        """
        Tests the case where `use_unknown` is True; this means that the OHE vector is N categories + 1, where the first index represents UNK token.

        Checks:
        (1) UNKS are handled by "1" in the first index
        (2) Nones or unrecognized categories are both handled
        (3) The decode/encode map order is the same
        """
        data = ['category 1', 'category 3', 'category 4', None, 'category 3']
        test_data = ['CATEGORY 1', 'category 2', 'category 1', 'category 3', None]

        ytest = [
            _UNCOMMON_WORD,
            _UNCOMMON_WORD,
            'category 1',
            'category 3',
            _UNCOMMON_WORD,
        ]

        enc = OneHotEncoder(use_unknown=True)
        enc.prepare(data)

        # Check the encoded patterns correct
        encoded_data = enc.encode(data)
        self.assertTrue(
            (
                encoded_data
                == Tensor(
                    [
                        [0.0, 1.0, 0.0, 0.0],  # category 1
                        [0.0, 0.0, 1.0, 0.0],  # category 3
                        [0.0, 0.0, 0.0, 1.0],  # category 4
                        [1.0, 0.0, 0.0, 0.0],  # None
                        [0.0, 0.0, 1.0, 0.0],  # category 3
                    ]
                )
            ).all()
        )

        # Check the decoded patterns correct
        decoded_data = enc.decode(enc.encode(test_data))
        for i in range(len(ytest)):
            self.assertTrue(decoded_data[i] == ytest[i])

    def test_check_probs_with_unknown(self):
        """ Check probability calculation """
        data = ['category 1', 'category 3', 'category 4', None, 'category 3']

        enc = OneHotEncoder(use_unknown=True)
        enc.prepare(data)

        # Make data to represent random weights that do not sum to 1
        torch.manual_seed(1)
        wt_vec = torch.rand(size=(len(data), len(enc.map)))

        _, probs, _ = enc.decode_probabilities(wt_vec)
        self.assertTrue(np.all([np.isclose(sum(i), 1) for i in probs]))

    def test_encode_and_decode_with_return_zeros(self):
        """
        Tests the case where `use_unknown` is False; this means that the OHE vector is N categories, where a vector of all 0s represents the unknown token

        Checks:
        (1) UNKS are handled by a 0 vector returned
        (2) Nones or unrecognized categories are both handled
        (3) The decode/encode map order is the same
        """

        data = ['category 1', 'category 3', 'category 4', None]
        test_data = ['category 2', 'category 1', 'category 3', None]

        enc = OneHotEncoder(use_unknown=False)
        enc.prepare(data)
        encoded_data = enc.encode(data)
        self.assertTrue(
            (
                encoded_data
                == Tensor(
                    [
                        [1.0, 0.0, 0.0],  # category 1
                        [0.0, 1.0, 0.0],  # category 3
                        [0.0, 0.0, 1.0],  # category 4
                        [0.0, 0.0, 0.0],  # None
                    ]
                )
            ).all()
        )

        decoded_data = enc.decode(enc.encode(test_data))
        self.assertEqual(
            decoded_data,
            [
                _UNCOMMON_WORD,  # category 2 is not seen, thus encoded to zero, thus decoded to uncommon word
                'category 1',  # seen before
                'category 3',  # seen before
                _UNCOMMON_WORD,  # None is encoded to zero, thus decoded to uncommon word
            ],
        )

    def test_check_probs_with_unknown(self):
        """ Check probability calculation where `use_unknown=False` """
        data = ['category 1', 'category 3', 'category 4', None, 'category 3']

        enc = OneHotEncoder(use_unknown=False)
        enc.prepare(data)

        # Make data to represent random weights that do not sum to 1
        torch.manual_seed(1)
        wt_vec = torch.rand(size=(len(data), len(enc.map)))

        _, probs, _ = enc.decode_probabilities(wt_vec)
        self.assertTrue(np.all([np.isclose(sum(i), 1) for i in probs]))

    def test_target_distro_scaled_to_1(self):
        """
        Check whether target distribution passed and handled properly
        """
        data = ["apple", "apple", "banana", "apple", "apple", "orange"]

        # Scaled weights (sum to 1)
        tweights = {"apple": 4 / 6, "banana": 1 / 6, "orange": 1 / 6}

        # Get the ground-truth inverse weights
        iweights = Tensor([1 / i for i in tweights.values()])

        enc = OneHotEncoder(use_unknown=False, is_target=True, target_weights=tweights)
        enc.prepare(data)

        # Check inverse weights correct
        self.assertTrue(np.all(((enc.inv_target_weights - iweights) == 0).tolist()))

    def test_distro_nonzeroweights(self):
        """
        Tests if target wts do not sum to 1 properly handled.

        This handles cases where people may choose something like 1/class_size for weights.

        Index-weight order is sensitive to the mapping
        """

        data = ["apple", "apple", "banana", "apple", "apple", "orange"]
        tweights = {"apple": 100, "orange": 5000, "banana": 4}

        enc = OneHotEncoder(use_unknown=False, is_target=True, target_weights=tweights)
        enc.prepare(data)

        # Get the ground-truth inverse weights
        iweights = torch.ones(size=(len(tweights), ))

        for key, value in tweights.items():
            iweights[enc.map[key]] = 1/value

        # Reorder the weights with regards to the map
        self.assertTrue(np.all(((enc.inv_target_weights - iweights) == 0).tolist()))

    def test_distro_zero(self):
        """ Tests edge cause where target weights have a 0 weight which is unacceptable for downstream processing (inverse weights will 1/0) """
        data = ["apple", "apple", "banana", "apple", "apple", "orange"]

        # Arbitrary weights (ex: number of examples)
        tweights = {"apple": 100,  "banana": 1, "orange": 0}

        enc = OneHotEncoder(use_unknown=False, is_target=True, target_weights=tweights)

        # Check if 0-weight class is rejected
        self.assertRaises(ValueError, enc.prepare, data)
