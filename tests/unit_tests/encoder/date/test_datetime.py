import unittest
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from lightwood.encoder.datetime.datetime import DatetimeEncoder
from lightwood.encoder.datetime.datetime_sin_normalizer import DatetimeNormalizerEncoder
np.random.seed(1)


class TestDatetimeEncoder(unittest.TestCase):
    @staticmethod
    def _create_timestamp():
        min_ts = pd.Timestamp.min
        max_ts = pd.Timestamp.max
        return np.random.randint(min_ts.timestamp(), max_ts.timestamp())

    def test_raise_encode_type(self):
        data = [1, 2, 3]
        enc = DatetimeEncoder()
        enc.prepare([])
        self.assertRaises(Exception, enc.encode, data)

    def test_encode(self):
        data = pd.Series([self._create_timestamp() for _ in range(10_000)])
        data[0] = np.nan
        enc = DatetimeEncoder()
        enc.prepare([])
        encoded_repr = enc.encode(data)
        assert not torch.isinf(encoded_repr).any()
        assert not torch.isnan(encoded_repr).any()

    def test_decode(self):
        data = pd.Series([self._create_timestamp() for _ in range(1_000)])
        data[0] = np.nan
        enc = DatetimeEncoder()
        enc.prepare([])
        encoded_repr = enc.encode(data)
        assert not torch.isinf(encoded_repr).any()
        assert not torch.isnan(encoded_repr).any()
        dec_data = enc.decode(encoded_repr)

        for d, t in zip(dec_data[1:], data.tolist()[1:]):
            # ignore edge cases within border of supported years
            if pd.Timestamp.min.year + 1 < datetime.fromtimestamp(t).year < pd.Timestamp.max.year - 1:
                if not np.isclose(d, t):
                    assert datetime.fromtimestamp(d) == datetime.fromtimestamp(t)
                else:
                    assert np.isclose(d, t)
        assert np.isnan(dec_data[0])

    def test_sinusoidal_encoding(self):
        data = [self._create_timestamp() for _ in range(100)]
        normalizer = DatetimeNormalizerEncoder(sinusoidal=True)
        normalizer.prepare([])

        results = normalizer.encode(data)
        null = np.full_like(results, 0.5)
        self.assertTrue(np.allclose(results, null, atol=0.5))  # every value in [0, 1]

        recons = normalizer.decode(results)
        for a, b in zip(recons, data):
            self.assertEqual(a, b)  # check correct reconstruction

    def test_cap_invalid_dates(self):
        """
        Test decoding robustness against invalid magnitudes in datetime encodings.
        """
        data = [self._create_timestamp() for _ in range(100)]
        limits = {
            'lower': {'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'corruption': -0.5},
            'upper': {'month': 12, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59, 'corruption': 1.5}
        }
        normalizer = DatetimeNormalizerEncoder()
        normalizer.prepare([])

        # change descriptor to invalid values in each dimension (out of 0-1 range)
        for limit in limits.values():
            for i, attr in zip(range(7), normalizer.fields):
                if attr in ('year', 'weekday'):
                    continue
                else:
                    vector = normalizer.encode(data)
                    vector[:, :, i] = limit['corruption']
                    recons = datetime.fromtimestamp(normalizer.decode(vector)[0])

                    # decoding correctly caps the invalid vector component
                    self.assertEqual(getattr(recons, attr), limit[attr])
