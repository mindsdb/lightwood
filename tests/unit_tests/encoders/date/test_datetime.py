import unittest
from lightwood.encoders.datetime.datetime import DatetimeEncoder


class TestDatetimeEncoder(unittest.TestCase):
    def test_decode(self):
        data = [1555943147, None, 1555943147]

        enc = DatetimeEncoder()
        enc.prepare([])
        enc.decode(enc.encode(data))

    def test_sinusoidal_encoding(self):
        import numpy as np
        from dateutil.parser import parse as parse_datetime

        dates = ['1971-12-1 00:01', '2000-5-29 23:59:30', '2262-3-11 3:0:5']
        dates = [parse_datetime(d) for d in dates]
        data = [d.timestamp() for d in dates]

        normalizer = DatetimeEncoder(sinusoidal=True)
        normalizer.prepare([])

        results = normalizer.encode(data)
        null = np.full_like(results, 0.5)
        self.assertTrue(np.allclose(results, null, atol=0.5))  # every value in [0, 1]

        recons = normalizer.decode(results)
        for a, b in zip(recons, data):
            self.assertEqual(a, b)  # check correct reconstruction
