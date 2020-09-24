import unittest
import random
import torch
from sklearn.metrics import r2_score
from lightwood.encoders.numeric import NumericEncoder
from lightwood.encoders.text import DistilBertEncoder
from lightwood import COLUMN_DATA_TYPES


class TestDistilBERT(unittest.TestCase):
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

        enc = DistilBertEncoder()

        enc.prepare(priming_data,
                            training_data={'targets': [
                                {'output_type': COLUMN_DATA_TYPES.NUMERIC,'encoded_output': encoded_data_1},
                                {'output_type': COLUMN_DATA_TYPES.NUMERIC, 'encoded_output': encoded_data_1}
                            ]})#

        encoded_predicted_target = enc.encode(test_data).tolist()

        predicted_targets_1 = output_1_encoder.decode(torch.tensor([x[:3] for x in encoded_predicted_target]))
        predicted_targets_2 = output_1_encoder.decode(torch.tensor([x[3:] for x in encoded_predicted_target]))

        for predicted_targets in [predicted_targets_1, predicted_targets_2]:
            real = list(test_target)
            pred = list(predicted_targets)

            # handle nan
            for i in range(len(pred)):
                try:
                    float(pred[i])
                except:
                    pred[i] = 0

            print(real[0:25], '\n', pred[0:25])
            encoder_accuracy = r2_score(real, pred)

            print(f'Categorial encoder accuracy for: {encoder_accuracy} on testing dataset')
            # assert(encoder_accuracy > 0.5)
