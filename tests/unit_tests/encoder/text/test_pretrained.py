import unittest
import random
import torch
from sklearn.metrics import r2_score
from lightwood.encoder.numeric import NumericEncoder
from lightwood.encoder.text import PretrainedLangEncoder
from lightwood.api.dtype import dtype
import pandas as pd


class TestPretrainedLangEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        random.seed(2)
        priming_data = []
        primting_target = []
        test_data = []
        test_target = []
        for i in range(0, 300):
            if random.randint(1, 5) == 3:
                test_data.append(str(i) + ''.join(['n'] * i))
                # test_data.append(str(i))
                test_target.append(i)
            # else:
            priming_data.append(str(i) + ''.join(['n'] * i))
            # priming_data.append(str(i))
            primting_target.append(i)

        output_1_encoder = NumericEncoder(is_target=True)
        output_1_encoder.prepare(primting_target)

        encoded_data_1 = output_1_encoder.encode(primting_target)
        encoded_data_1 = encoded_data_1.tolist()

        enc = PretrainedLangEncoder(stop_after=10)

        enc.prepare(pd.Series(priming_data), pd.Series(priming_data),
                    encoded_target_values={'targets': [
                        {'output_type': dtype.float, 'encoded_output': encoded_data_1},
                    ]})

        encoded_predicted_target = enc.encode(test_data).tolist()

        predicted_targets_1 = output_1_encoder.decode(torch.tensor([x[:3] for x in encoded_predicted_target]))

        for predicted_targets in [predicted_targets_1]:
            real = list(test_target)
            pred = list(predicted_targets)

            # handle nan
            for i in range(len(pred)):
                try:
                    float(pred[i])
                except Exception:
                    pred[i] = 0

            print(real[0:25], '\n', pred[0:25])
            encoder_accuracy = r2_score(real, pred)

            print(f'Categorial encoder accuracy for: {encoder_accuracy} on testing dataset')
            # assert(encoder_accuracy > 0.5)
