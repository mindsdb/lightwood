import unittest
import numpy as np
import pandas as pd
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
        data = pd.Series([1, 1.1, 2, -8.6, None, 0])

        encoder = NumericEncoder()
        encoder.prepare(data)
        encoded_vals = encoder.encode(data)

        # sign component check
        self.assertTrue(encoded_vals[0][0] > 0)
        self.assertTrue(encoded_vals[1][0] > 0)
        self.assertTrue(encoded_vals[2][0] > 0)
        self.assertTrue(encoded_vals[3][0] == 0)

        # none component check
        for i in range(0, len(encoded_vals)):
            if i != 4:
                self.assertTrue(encoded_vals[i][-1] == 0)
            else:
                self.assertTrue(encoded_vals[i][-1] == 1)

        # exp component nan edge case check
        self.assertTrue(encoded_vals[4][2] == 0)

        # compare decoded v/s real
        decoded_vals = encoder.decode(encoded_vals)
        for decoded, real in zip(decoded_vals, data.tolist()):
            if decoded is None:
                self.assertTrue((real is None) or (real != real))
            else:
                np.testing.assert_almost_equal(round(decoded, 6), round(real, 6))

    def test_positive_domain(self):
        data = pd.Series([-1, -2, -100, 5, 10, 15])
        for encoder in [NumericEncoder(), TsNumericEncoder()]:
            encoder.is_target = True  # only affects target values
            encoder.positive_domain = True
            encoder.prepare(data)
            decoded_vals = encoder.decode(encoder.encode(data))

            for val in decoded_vals:
                self.assertTrue(val >= 0)

    def test_log_overflow_and_none(self):
        data = pd.Series(list(range(-2000, 2000, 66)))
        encoder = NumericEncoder()

        encoder.is_target = True
        encoder.positive_domain = True
        encoder.decode_log = True
        encoder.prepare(data)
        encoder.decode(encoder.encode(data))

        for i in range(0, 70, 10):
            encoder.decode(torch.Tensor([[0, pow(2, i), 0]]))

    def test_nan_encoding(self):
        # Generate some numbers
        data = list(range(-50, 50, 2))

        # Add invalid values to the data
        invalid_data = _pollute(data)

        # Prepare with the correct data and decode invalid data
        encoder = NumericEncoder()
        encoder.prepare(pd.Series(data))
        for array in invalid_data:
            # Make sure the encoding has no nans or infs
            encoded_repr = encoder.encode(pd.Series(array))
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
            encoder.prepare(pd.Series(array))

            # Make sure the encoding has no nans or infs
            encoded_repr = encoder.encode(pd.Series(array))
            assert not torch.isnan(encoded_repr).any()
            assert not torch.isinf(encoded_repr).any()

            # Make sure the invalid value is decoded as `None` and the rest as numbers
            decoded_repr = encoder.decode(encoded_repr)
            for dec, real in zip(decoded_repr, array):
                if is_none(real):
                    assert is_none(dec)
                else:
                    assert not is_none(x) or x != 0.0

    def test_weights(self):
        data = np.random.normal(loc=0.0, scale=1.0, size=1000)
        hist, bin_edges = np.histogram(data, bins=10, density=False)

        # constrict bins so that final histograms align, throw out minimum bin as the np.searchsorted is left justified.

        bin_edges = bin_edges[1:]

        # construct target weight mapping. This mapping will round each entry to the lower bin edge.
        target_weights = {bin_edge: bin_edge for bin_edge in bin_edges}
        self.assertTrue(type(target_weights) is dict)

        # apply weight mapping
        encoder = NumericEncoder(target_weights=target_weights)
        generated_weights = encoder.get_weights(label_data=data)

        self.assertTrue(type(generated_weights) is np.ndarray)

        # distributions should match
        gen_hist, _ = np.histogram(generated_weights, bins=10, density=False)

        self.assertTrue(np.all(np.equal(hist, gen_hist)))
