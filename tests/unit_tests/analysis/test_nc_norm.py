import unittest
import numpy as np
from lightwood.analysis.nc.norm import Normalizer


class TestNcNormalizer(unittest.TestCase):
    def test_compute_numerical_labels(self):
        preds = np.array([1, 2.1, 3.2, -2, 1e10, 0])
        truths = np.array([1, 2, 3, 4, 5, 0])
        bounds = [0.5, 1.5]

        labels = Normalizer.compute_numerical_labels(preds, truths, bounds)
        self.assertTrue(np.min(labels) >= bounds[0])
        self.assertTrue(np.max(labels) <= bounds[1])
        self.assertFalse(np.isnan(labels).any())
        self.assertFalse(np.isinf(labels).any())

    def test_compute_categorical_labels(self):
        preds = np.array([[0.01, 0.99], [1, 0], [0.99, 0.1], [0, 1]])
        truths = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])

        labels = Normalizer.compute_categorical_labels(preds, truths)
        self.assertTrue(np.min(labels) >= 0)
        self.assertFalse(np.isnan(labels).any())
        self.assertFalse(np.isinf(labels).any())
