import math
import unittest
from lightwood.encoders.time_series import TsFreshTsEncoder

class TestTsFreshTs(unittest.TestCase):
    def test_encode(self):
        data = [' '.join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

        enc = TsFreshTsEncoder()
        enc.prepare_encoder(data)
        ret = enc.encode(data)

        self.assertTrue(len(ret) == len(data))
        self.assertTrue(len(ret) < 60)
