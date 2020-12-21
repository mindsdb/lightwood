import unittest
import numpy as np
from lightwood.encoders.time_series.helpers.common import *


class TestTimeSeriesHelpers(unittest.TestCase):
    def test_get_chunk(self):
        # dimensions: (batch_size, sequence_length, feature_dimension)
        start, step = 0, 2
        batch_size, seq_len, feat_dim = 10, 5, 2

        input_tensor = torch.rand(batch_size, seq_len, feat_dim)
        len_tensor = torch.zeros(batch_size).fill_(seq_len)
        data, target, lengths = get_chunk(input_tensor, len_tensor, start, step)

        # check length vector is correct
        assert lengths.shape[0] == batch_size
        assert lengths.numpy()[0] == seq_len-1

        # check data and target
        chunk_size = min(start + step, batch_size) - start
        assert data.shape == (batch_size, chunk_size, feat_dim)
        assert target.shape == (batch_size, chunk_size, feat_dim)
        assert torch.equal(data[:, 1, :], target[:, 0, :])

        # check edge case: correct cutoff at end of sequence
        start, step = 2, 4
        chunk_size = min(start + step, seq_len) - start - 1  # -1 because of cut off
        data, target, lengths = get_chunk(input_tensor, len_tensor, start, step)
        assert data.shape == (batch_size, chunk_size, feat_dim)
        assert target.shape == (batch_size, chunk_size, feat_dim)

    def test_mask(self):
        series = [1, 3, 2, 4]
        target = [
            [True, False, False, False],
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ]

        target = torch.tensor(target, dtype=torch.bool)
        series = torch.tensor(series)
        result = len_to_mask(series, zeros=False)
        self.assertTrue((result == target).all())
        target = ~target
        result = len_to_mask(series, zeros=True)
        self.assertTrue((result == target).all())

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
        self.assertTrue(encoded_target, encoded.tolist())
        dec = normalizer.decode(encoded)
        self.assertTrue(dec[-1][-1] == normalizer.unk)
        dec[-1][-1] = None
        self.assertTrue(data == dec)
