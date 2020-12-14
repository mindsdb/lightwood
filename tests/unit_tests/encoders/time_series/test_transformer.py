import copy
import logging
import torch
import unittest

from lightwood.helpers.device import get_devices
from lightwood.encoders.time_series import TransformerEncoder
from lightwood.encoders.time_series.helpers.transformer_helpers import len_to_mask


class TestTransformerEncoder(unittest.TestCase):
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
        params = {"encoded_vector_size": 16, "train_iters": 10, "learning_rate": 0.001}

        # Test sequences of different length
        # We just test the nothrow condition, as the control flow for BPTT and the normal one is the same
        # and the flow is tested in the next test
        data = [[1, 2, 3, 4, 5], [2, 3, 4], [3, 4, 5, 6]]
        encoder = TransformerEncoder(**params)
        encoder.prepare_encoder(data, feedback_hoop_function=print)

        # Test TBPTT. Training on this woudld require a better tuning of the lr and maybe a scheduler
        # Again, just test nothrow
        data = [
            torch.rand(torch.randint(low=5, high=120, size=(1,)).item()).tolist()
            for _ in range(87)
        ]
        encoder = TransformerEncoder(**params)
        encoder.prepare_encoder(data, feedback_hoop_function=print)

        # Test Overfit
        data = [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]]
        example = copy.deepcopy(data)
        params["train_iters"] = 1000
        encoder = TransformerEncoder(**params)
        encoder.prepare_encoder(data, feedback_hoop_function=print)

        # Test data
        example = torch.tensor(example)
        correct_answer = example[:, 1:]
        # Decoder overfit, discard last element as it doesn't correspond to answer
        answer = torch.tensor(encoder.encode(example))[:, :-1]
        # Round answer
        answer = answer.float().round()
        n = correct_answer.numel()
        correct = (correct_answer == answer).sum()
        print(
            "Decoder got {equal} correct and {unequal} incorrect".format(
                equal=correct, unequal=n - correct
            )
        )
        self.assertGreaterEqual(correct * 2, n - correct)
