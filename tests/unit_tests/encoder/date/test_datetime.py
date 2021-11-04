import unittest
from datetime import datetime
import numpy as np
from dateutil.parser import parse as parse_datetime
import torch
from lightwood.encoder.datetime.datetime import DatetimeEncoder
from lightwood.encoder.datetime.datetime_sin_normalizer import DatetimeNormalizerEncoder


class TestDatetimeEncoder(unittest.TestCase):
    def test_decode(self):
        data = [1555943147, None, 1555943147, '', np.nan]

        enc = DatetimeEncoder()
        enc.prepare([])
        encoded_repr = enc.encode(data)
        assert not torch.isinf(encoded_repr).any()
        assert not torch.isnan(encoded_repr).any()
        dec_data = enc.decode(encoded_repr)
        for d in dec_data:
            assert d in data

    def test_sinusoidal_encoding(self):
        dates = ['1971-12-1 00:01', '2000-5-29 23:59:30', '2262-3-11 3:0:5']
        dates = [parse_datetime(d) for d in dates]
        data = [d.timestamp() for d in dates]

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
        dates = [parse_datetime('2020-10-10 10:10:10')]
        data = [d.timestamp() for d in dates]
        limits = {
            'lower': {'month': 1, 'day': 1, 'hour': 0, 'minute': 0, 'second': 0, 'corruption': -0.5},
            'upper': {'month': 12, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59, 'corruption': 1.5}
        }

        normalizer = DatetimeNormalizerEncoder()
        normalizer.prepare([])

        # change descriptor to invalid values in each dimension (out of 0-1 range)
        for limit in limits.values():
            print(limit)
            for i, attr in zip(range(7), normalizer.fields):
                if attr in ('year', 'weekday'):
                    continue
                else:
                    vector = normalizer.encode(data)
                    vector[:, :, i] = limit['corruption']
                    recons = datetime.fromtimestamp(normalizer.decode(vector)[0])

                    # decoding correctly caps the invalid vector component
                    self.assertEqual(getattr(recons, attr), limit[attr])
