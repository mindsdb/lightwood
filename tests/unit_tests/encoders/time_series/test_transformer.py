import copy
import logging
import torch
import unittest

from lightwood.helpers.device import get_devices
from lightwood.encoders.time_series import TimeSeriesEncoder
from lightwood.encoders.time_series.helpers.transformer_helpers import TransformerEncoder, len_to_mask, get_chunk


class TestTransformerEncoder(unittest.TestCase):
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

    def test_overfit(self):
        logging.basicConfig(level=logging.DEBUG)
        params = {"encoded_vector_size": 16, "train_iters": 10, "learning_rate": 0.001,
                  "encoder_class": TransformerEncoder}

        data = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]] * 10000
        timesteps = len(data[0])
        example = copy.deepcopy(data)
        params["train_iters"] = 10
        encoder = TimeSeriesEncoder(**params)
        encoder._transformer_hidden_size = 32
        encoder.prepare(data, feedback_hoop_function=print)

        correct_answer = torch.tensor(example)[:, 1:]

        # discard last element as it doesn't correspond to answer
        data, lens = encoder._prepare_raw_data(torch.tensor(example)[:, :-1])
        data = torch.stack([d for d in data]).unsqueeze(-1).transpose(0, 1).to(encoder.device)
        output, hidden = encoder._encoder.forward(data, lens, data.device)

        assert hidden.shape == (timesteps - 1, len(example), encoder._transformer_hidden_size)
        assert output.shape == (timesteps - 1, len(example), 1)

        answer = output.transpose(0, 1).squeeze(-1)
        assert answer.shape == correct_answer.shape

        # check reconstruction
        correct_answer = correct_answer.to(dtype=answer.dtype, device=answer.device)
        results = torch.isclose(answer, correct_answer, atol=1)
        acc = (results.sum() / results.numel()).item()

        print(f'Transformer correctly reconstructed {round(100*acc, 2)}%')
        assert acc >= 0.5
