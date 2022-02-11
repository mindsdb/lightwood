import unittest
import numpy as np
import torch
from lightwood.encoder.time_series.rnn import TimeSeriesEncoder
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
import pandas as pd


class TestRnnEncoder(unittest.TestCase):

    def test_minmax_normalizer(self):
        data = [[-100.0, -5.0, 0.0, 5.0, 100.0],
                [-1000.0, -50.0, 0.0, 50.0, 1000.0],
                [-500.0, -405.0, -400.0, -395.0, -300.0],
                [300.0, 395.0, 400.0, 405.0, 500.0],
                [0.0, 1e3, 1e6, 1e9, 1e12]]
        normalizer = MinMaxNormalizer()
        normalizer.prepare(data)
        reconstructed = normalizer.decode(normalizer.encode(data))
        self.assertTrue(np.allclose(data, reconstructed, atol=0.1))

    def test_cat_normalizer(self):
        data = [['a', 'b', 'c'],
                ['c', 'b', 'b'],
                ['a', 'a', None]]
        encoded_target = [[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                          [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
                          [[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]]
        normalizer = CatNormalizer()
        normalizer.prepare(data)
        encoded = normalizer.encode(data)
        self.assertTrue(encoded_target, encoded)
        dec = [normalizer.decode(e).squeeze().tolist() for e in encoded]
        self.assertTrue(dec[-1][-1] == normalizer.unk)
        dec[-1][-1] = None
        self.assertTrue(data == dec)

    def test_overfit(self):
        series = [[1, 2, 3, 4, 5, 6],
                  [2, 3, 4, 5, 6, 7],
                  [3, 4, 5, 6, 7, 8],
                  [4, 5, 6, 7, 8, 9]]

        example = ([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],  # query
                   [4, 5, 6, 7])                                    # answer

        data = series * 50
        timesteps = 6
        batch_size = 1

        encoder = TimeSeriesEncoder(stop_after=10)
        encoder.prepare(pd.Series(data), pd.Series(data),
                        feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
        encoded = encoder.encode(data)
        decoded = encoder.decode(encoded, steps=timesteps).tolist()

        equal = 0
        unequal = 0
        self.assertEqual(len(data), len(decoded))
        self.assertEqual(len(data[0]), len(decoded[0]))

        for i in range(len(data)):
            for t in range(timesteps):
                if round(decoded[i][0][t], 0) == round(data[i][0][t], 0):
                    equal += 1
                else:
                    unequal += 1

        print(f'Decoder got {equal} correct and {unequal} incorrect')
        self.assertGreaterEqual(equal * 2, unequal)

        error_margin = 10  # 3
        query, answer = example
        encoded_data, preds = encoder.encode(query, get_next_count=1)
        decoded_data = encoder.decode(encoded_data, steps=3).tolist()

        # check reconstruction
        float_query = [list(map(float, q)) for q in query[0]]
        for qry, dec in zip(float_query, decoded_data[0]):
            for truth, pred in zip(qry, dec):
                self.assertGreater(error_margin, abs(truth - pred))

        # check next value prediction
        preds = torch.reshape(preds, (1, -1)).tolist()[-1]
        for ans, pred in zip(answer, preds):
            self.assertGreater(error_margin, abs(pred - ans))
