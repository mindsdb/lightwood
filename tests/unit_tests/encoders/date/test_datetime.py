import unittest
from lightwood.encoders.datetime.datetime import DatetimeEncoder


class TestDatetimeEncoder(unittest.TestCase):
    def test_decode(self):
        data = [1555943147, None, 1555943147]

        enc = DatetimeEncoder()
        enc.prepare_encoder([])
        enc.decode(enc.encode(data))
