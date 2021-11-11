import unittest
import numpy as np
import torch
from lightwood.encoder.numeric import NumericEncoder
from lightwood.encoder.numeric import TsNumericEncoder
from lightwood.helpers.general import is_none


def _pollute(array):
    return [
        array + [np.nan],
        array + [np.inf],
        array + [None]
    ]


class TestNumericEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        data = [1, 1.1, 2, -8.6, None, 0]

        encoder = NumericEncoder()

        encoder.prepare(data)
        encoded_vals = encoder.encode(data)

        self.assertTrue(encoded_vals[1][1] > 0)
        self.assertTrue(encoded_vals[2][1] > 0)
        self.assertTrue(encoded_vals[3][1] > 0)
        for i in range(0, 3):
            self.assertTrue(encoded_vals[i][2] == 0)
        self.assertTrue(encoded_vals[3][2] == 1)
        self.assertTrue(encoded_vals[4][3] == 0)

        decoded_vals = encoder.decode(encoded_vals)

        for i in range(len(encoded_vals)):
            if decoded_vals[i] is None:
                self.assertTrue(decoded_vals[i] == data[i])
            else:
                np.testing.assert_almost_equal(round(decoded_vals[i], 10), round(data[i], 10))

    def test_positive_domain(self):
        data = [-1, -2, -100, 5, 10, 15]
        for encoder in [NumericEncoder(), TsNumericEncoder()]:
            encoder.is_target = True        # only affects target values
            encoder.positive_domain = True
            encoder.prepare(data)
            decoded_vals = encoder.decode(encoder.encode(data))

            for val in decoded_vals:
                self.assertTrue(val >= 0)

    def test_log_overflow_and_none(self):
        data = list(range(-2000, 2000, 66))
        encoder = NumericEncoder()

        encoder.is_target = True
        encoder.positive_domain = True
        encoder.decode_log = True
        encoder.prepare(data)
        encoder.decode(encoder.encode(data))

        for i in range(0, 70, 10):
            encoder.decode([[0, pow(2, i), 0]])

    def test_nan_encoding(self):
        # Generate some numbers
        data = list(range(-50, 50, 2))

        # Add invalid values to the data
        invalid_data = _pollute(data)

        # Prepare with the correct data and decode invalid data
        encoder = NumericEncoder()
        encoder.prepare(data)
        for array in invalid_data:
            # Make sure the encoding has no nans or infs
            encoded_repr = encoder.encode(array)
            assert not torch.isnan(encoded_repr).any()
            assert not torch.isinf(encoded_repr).any()

            # Make sure the invalid value is decoded as `None` and the rest as numbers
            decoded_repr = encoder.decode(encoded_repr)
            for x in decoded_repr[:-1]:
                assert not is_none(x)
            assert decoded_repr[-1] is None

        # Prepare with the invalid data and decode the valid data
        for array in invalid_data:
            encoder = NumericEncoder()
            encoder.prepare(array)

            # Make sure the encoding has no nans or infs
            encoded_repr = encoder.encode(data)
            assert not torch.isnan(encoded_repr).any()
            assert not torch.isinf(encoded_repr).any()

            # Make sure the invalid value is decoded as `None` and the rest as numbers
            decoded_repr = encoder.decode(encoded_repr)
            for x in decoded_repr:
                assert not is_none(x)

        # Prepare with the invalid data and decode invalid data
        for array in invalid_data:
            encoder = NumericEncoder()
            encoder.prepare(array)
            # Make sure the encoding has no nans or infs
            encoded_repr = encoder.encode(array)
            assert not torch.isnan(encoded_repr).any()
            assert not torch.isinf(encoded_repr).any()

            # Make sure the invalid value is decoded as `None` and the rest as numbers
            decoded_repr = encoder.decode(encoded_repr)
            for x in decoded_repr[:-1]:
                assert not is_none(x)
            assert decoded_repr[-1] is None
