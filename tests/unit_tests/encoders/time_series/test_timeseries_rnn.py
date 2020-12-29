import unittest

from lightwood.encoders.time_series import TimeSeriesEncoder
from lightwood.encoders.time_series.helpers.rnn_helpers import *


class TestRnnEncoder(unittest.TestCase):
    def test_overfit(self):
        series = [[1, 2, 3, 4, 5, 6],
                  [2, 3, 4, 5, 6, 7],
                  [3, 4, 5, 6, 7, 8],
                  [4, 5, 6, 7, 8, 9]]

        example = ([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],  # query
                   [4, 5, 6, 7])                                  # answer

        data = series * 100
        timesteps = 6
        batch_size = 1

        encoder = TimeSeriesEncoder(encoded_vector_size=15, train_iters=3, encoder_class=EncoderRNNNumerical)
        encoder.prepare(data, feedback_hoop_function=lambda x: print(x), batch_size=batch_size)
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
        self.assertGreaterEqual(equal*2, unequal)

        error_margin = 10 # 3
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
