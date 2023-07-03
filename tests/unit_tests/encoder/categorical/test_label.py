import unittest
from torch import Tensor
import pandas as pd
from lightwood.encoder.categorical.simple_label import (
    SimpleLabelEncoder,
)
from lightwood.helpers.constants import _UNCOMMON_WORD


class TestLabel(unittest.TestCase):
    """ Test the label encoder """

    def test_encode_and_decode(self):
        """
        Tests encoder end to end

        Checks:
        (1) UNKS are assigned to 0th index
        (2) Nones or unrecognized categories are both handled
        (3) The decode/encode map order is the same
        """ # noqa
        data = pd.Series(['category 1', 'category 3', 'category 4', None, 'category 3'])
        test_data = pd.Series(['unseen', 'category 4', 'category 1', 'category 3', None])
        n_points = data.nunique()

        ytest = [
            _UNCOMMON_WORD,
            'category 4',
            'category 1',
            'category 3',
            _UNCOMMON_WORD,
        ]

        enc = SimpleLabelEncoder()
        enc.prepare(data)

        # Check the encoded patterns correct
        encoded_data = enc.encode(data)
        print(encoded_data)
        self.assertTrue(
            (
                encoded_data
                == Tensor(
                    [
                        1 / n_points,  # category 1
                        2 / n_points,  # category 3
                        3 / n_points,  # category 4
                        0 / n_points,  # None
                        2 / n_points,  # category 3
                    ]
                ).reshape(-1, 1)
            ).all()
        )

        # Check the decoded patterns correct
        decoded_data = enc.decode(enc.encode(test_data))
        print(decoded_data)
        for i in range(len(ytest)):
            self.assertTrue(decoded_data[i] == ytest[i])
