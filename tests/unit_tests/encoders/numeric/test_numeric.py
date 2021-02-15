import unittest
import numpy as np
from lightwood.encoders.numeric import NumericEncoder


class TestNumericEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        data = [1,1.1,2,-8.6,None,0]

        encoder = NumericEncoder()

        encoder.prepare(data)
        encoded_vals = encoder.encode(data)

        self.assertTrue(encoded_vals[1][1] > 0)
        self.assertTrue(encoded_vals[2][1] > 0)
        self.assertTrue(encoded_vals[3][1] > 0)
        for i in range(0,3):
            self.assertTrue(encoded_vals[i][2] == 0)
        self.assertTrue(encoded_vals[3][2] == 1)
        self.assertTrue(encoded_vals[4][3] == 0)

        decoded_vals = encoder.decode(encoded_vals)

        for i in range(len(encoded_vals)):
            if decoded_vals[i] is None:
                self.assertTrue(decoded_vals[i] == data[i])
            else:
                np.testing.assert_almost_equal(round(decoded_vals[i],10), round(data[i],10))

    def test_positive_domain(self):
        data = [-1, -2, -100, 5, 10, 15]
        encoder = NumericEncoder()

        encoder.is_target = True        # only affects target values
        encoder.positive_domain = True
        encoder.prepare(data)
        decoded_vals = encoder.decode(encoder.encode(data))

        for val in decoded_vals:
            self.assertTrue(val >= 0)

    def test_log_overflow_and_none(self):
        data = list(range(-2000,2000,66))
        data.extend([None] * 200)
        encoder = NumericEncoder()

        encoder.is_target = True
        encoder.positive_domain = True
        encoder.decode_log = True
        encoder.prepare(data)
        decoded_vals = encoder.decode(encoder.encode(data))

        for i in range(0,70,10):
            encoder.decode([[0, pow(2,i), 0]])
