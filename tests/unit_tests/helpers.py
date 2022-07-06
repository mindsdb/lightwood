import unittest

import numpy as np
import pandas as pd

from lightwood.helpers.ts import Differencer


class TestTSDifferencer(unittest.TestCase):
    def test_numerical(self):
        D = Differencer()
        D.fit(np.array([1, 2, 3]))

        reconstructed = D.inverse_transform(D.transform(np.array([4, 5, 6])))
        target = pd.Series([3, 4, 5])
        self.assertTrue(reconstructed.all() == target.all())

        reconstructed = D.inverse_transform(D.transform(np.array([4, 5, 6])), init=4)
        target = pd.Series([4, 5, 6])
        self.assertTrue(reconstructed.all() == target.all())
