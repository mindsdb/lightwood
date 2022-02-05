import unittest

import numpy as np
import pandas as pd

from lightwood.data.timeseries_analyzer import get_naive_residuals
from lightwood.helpers.general import mase, evaluate_array_accuracy


class TestTransformTS(unittest.TestCase):
    def test_mase(self):
        true = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # edge case: perfect forecast
        for scale_error in [1e0, 1e2, 1e4]:
            self.assertTrue(mase(true, true, scale_error, fh=5) == 0)

        # check naive forecast is exactly one
        naive_residual = np.average(abs(true[:, 1:] - true[:, :-1]))
        self.assertTrue(mase(true[:, 1:], true[:, :-1], naive_residual, fh=4) == 1)

        # edge case: constant series
        true = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]])
        pred = np.array([[4.0, 4.0, 4.0, 4.0, 4.0]])
        self.assertTrue(mase(true, pred, 0.0, fh=5) == 2.0)

        # test multiple instance handling (i.e. two 5-step-ahead forecasts)
        true = [[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]]
        pred = [[15, 25, 35, 45, 55], [65, 75, 85, 95, 105]]
        self.assertTrue(mase(true, pred, scale_error=5, fh=5) == 1)
        self.assertTrue(mase(true, pred, scale_error=1, fh=5) == 5)
        self.assertTrue(mase(true, pred, scale_error=10, fh=5) == 0.5)

    def test_get_residuals(self):
        data_len = 10

        target = [i for i in range(data_len)]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [1.0 for _ in range(data_len - 1)])
        self.assertEqual(mean, 1)

        target = [0 for _ in range(data_len)]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [0.0 for _ in range(data_len - 1)])
        self.assertEqual(mean, 0)

        target = [1, 4, 2, 5, 3]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [3.0, 2.0, 3.0, 2.0])
        self.assertEqual(mean, 2.5)

    def test_evaluate_array_r2_accuracy(self):
        true = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])
        self.assertTrue(evaluate_array_accuracy(true, true) == 1.0)

        pred = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.assertTrue(evaluate_array_accuracy(true, pred) == 0.0)

        pred = np.array([[i + 1 for i in instance] for instance in true])
        self.assertGreaterEqual(evaluate_array_accuracy(true, pred), 0.99)

        pred = np.array([[i - 1 for i in instance] for instance in true])
        self.assertGreaterEqual(evaluate_array_accuracy(true, pred), 0.99)

        pred = np.array([[-i for i in instance] for instance in true])
        self.assertTrue(evaluate_array_accuracy(true, pred) == 0.0)
