import copy
import logging
import torch
import unittest

from lightwood.encoders.time_series import TimeSeriesEncoder
from lightwood.encoders.time_series.helpers.transformer_helpers import TransformerEncoder


class TestTransformerEncoder(unittest.TestCase):
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
