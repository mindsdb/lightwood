import unittest
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.encoders.time_series.time_series import TimeSeriesEncoder
from lightwood.encoders.time_series.helpers.cnn_helpers import EncoderCNNts


def simple_data_generator(length, dims):
    return [[20*i + j for j in range(length)] for i in range(dims)]


def nonlin_data_generator(length, dims):
    # y = x^3 + x^2 + c
    return [[j**3 + j**2 + i for j in range(length)] for i in range(dims)]


class TestCnnEncoder(unittest.TestCase):

    def test_linear_overfit(self):
        n_dims = 1
        length = 400
        data = simple_data_generator(length, n_dims)
        data = 200*data
        batch_size = 32

        self._train_epochs = 40
        self._learning_rate = 0.005

        self._blocks = [32, 16, 8, 1]
        self._kernel_size = 5

        encoder = TimeSeriesEncoder(ts_n_dims=n_dims, encoder_class=EncoderCNNts)
        encoder.original_type = COLUMN_DATA_TYPES.NUMERIC
        final_loss = encoder.prepare(data, feedback_hoop_function=lambda x: print(x), batch_size=batch_size)

        encoded = encoder.encode(data[0])  # encode one sample
        decoded = encoder.decode(encoded)
        assert final_loss < 1
        assert encoded.shape == (1, 2, 4)
        assert decoded.shape == (5, 10, 15)

    def test_nonlinear_overfit(self):
        n_dims = 1
        length = 800
        data = nonlin_data_generator(length, n_dims)
        data = 500*data
        batch_size = 32

        self._train_epochs = 50
        self._learning_rate = 0.005

        self._blocks = [128, 64, 32, 8, 2]
        self._kernel_size = 5

        import torch
        print(torch.Tensor(data).shape)

        encoder = TimeSeriesEncoder(ts_n_dims=n_dims, encoder_class=EncoderCNNts)
        final_loss = encoder.prepare(data, feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
        return final_loss
