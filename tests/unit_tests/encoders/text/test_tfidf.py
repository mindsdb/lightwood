import unittest
import random
import string
from lightwood.encoders.text import TfidfEncoder


class TestTfidfEncoder(unittest.TestCase):
    def test_encode(self):
        random.seed(2)
        text = [''.join(random.choices(string.printable, k=random.randint(5,500))) for x in range(1000)]

        enc = TfidfEncoder()
        enc.prepare(text)
        encoded_data = enc.encode(text)
        print(encoded_data)
