import unittest

import numpy as np
import pandas as pd

from lightwood.data.timeseries_analyzer import get_naive_residuals
from lightwood.helpers.general import evaluate_array_accuracy


class TestTransformTS(unittest.TestCase):
    def test_get_residuals(self):
        data_len = 10

        target = [i for i in range(data_len)]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [i for i in range(1, data_len)])
        self.assertEqual(mean, np.mean([i for i in range(1, data_len)]))

        target = [0 for _ in range(data_len)]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [0.0 for _ in range(data_len - 1)])
        self.assertEqual(mean, 0)

        target = [1, 4, 2, 5, 3]
        all_residuals, mean = get_naive_residuals(pd.DataFrame(target))
        self.assertEqual(all_residuals, [3.0, 1.0, 4.0, 2.0])
        self.assertEqual(mean, np.mean([3.0, 1.0, 4.0, 2.0]))

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
