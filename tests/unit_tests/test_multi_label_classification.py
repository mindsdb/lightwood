import torch
import numpy as np
import unittest
import pandas as pd
import random
import string

from sklearn.metrics import f1_score, r2_score

from lightwood import Predictor
from lightwood.constants.lightwood import ColumnDataTypes


class TestMultiLabelPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(66)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(66)
        random.seed(66)

    def get_vocab(self, n_categories):
        return {i: ''.join(random.choices(string.ascii_uppercase, k=5)) for i in range(n_categories)}

    def test_multiple_categories_as_input(self):
        vocab = self.get_vocab(10)
        # tags contains up to 2 randomly selected tags
        # y contains the sum of indices of tags
        # the dataset should be nearly perfectly predicted
        n_points = 10000
        tags = []
        y = []
        for i in range(n_points):
            row_tags = []
            row_y = 0
            for k in range(2):
                if random.random() > 0.2:
                    selected_index = random.randint(0, len(vocab)-1)
                    if vocab[selected_index] not in row_tags:
                        row_tags.append(vocab[selected_index])
                        row_y += selected_index
            tags.append(row_tags)
            y.append(row_y)

        df = pd.DataFrame({'tags': tags, 'y': y})

        config = {
            'input_features': [
                {'name': 'tags', 'type': ColumnDataTypes.MULTIPLE_CATEGORICAL}
            ],
            'output_features': [
                {'name': 'y', 'type': ColumnDataTypes.NUMERIC}
            ],
        }
        df_train = df.iloc[:round(n_points * 0.9)]
        df_test = df.iloc[round(n_points * 0.9):]

        predictor = Predictor(config)

        predictor.learn(from_data=df_train,
                        stop_training_after_seconds=10)

        predictions = predictor.predict(when_data=df_test)

        test_y = df_test.y
        predicted_y = predictions['y']['predictions']

        score = r2_score(test_y, predicted_y)
        print('Test R2 score', score)
        # The score check is very light because we only allow the model to train for a few seconds
        # We are just checking that it learns something and predicts properly, not benchmarking here
        self.assertGreaterEqual(score, 0.15)

    def test_multiple_categories_as_output(self):
        vocab = self.get_vocab(10)
        # x1 contains the index of first tag present
        # x2 contains the index of second tag present
        # if a tag is missing then x1/x2 contain -1 instead
        # Thus the dataset should be perfectly predicted
        n_points = 10000
        x1 = [random.randint(0, len(vocab)-1) if random.random() > 0.2 else -1 for i in range(n_points)]
        x2 = [random.randint(0, len(vocab)-1) if random.random() > 0.2 else -1 for i in range(n_points)]
        tags = []
        for x1_index, x2_index in zip(x1, x2):
            row_tags = set([vocab.get(x1_index), vocab.get(x2_index)])
            row_tags = [x for x in row_tags if x is not None]
            tags.append(row_tags)

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'tags': tags})

        config = {
            'input_features': [
                {'name': 'x1',
                 'type': ColumnDataTypes.CATEGORICAL,
                 },
                {'name': 'x2',
                 'type': ColumnDataTypes.CATEGORICAL,
                 },
            ],
            'output_features': [
                {'name': 'tags', 'type': ColumnDataTypes.MULTIPLE_CATEGORICAL}
            ],
        }
        df_train = df.iloc[:round(n_points*0.9)]
        df_test = df.iloc[round(n_points*0.9):]

        predictor = Predictor(config)

        predictor.learn(from_data=df_train,
                        stop_training_after_seconds=10)

        predictions = predictor.predict(when_data=df_train)
        train_tags = df_train.tags
        predicted_tags = predictions['tags']['predictions']
        train_tags_encoded = predictor._mixer.encoders['tags'].encode(train_tags)
        pred_labels_encoded = predictor._mixer.encoders['tags'].encode(predicted_tags)
        score = f1_score(train_tags_encoded, pred_labels_encoded, average='weighted')
        print('Train f1 score', score)
        self.assertGreaterEqual(score, 0.15)

        predictions = predictor.predict(when_data=df_test)

        test_tags = df_test.tags
        predicted_tags = predictions['tags']['predictions']

        test_tags_encoded = predictor._mixer.encoders['tags'].encode(test_tags)
        pred_labels_encoded = predictor._mixer.encoders['tags'].encode(predicted_tags)
        score = f1_score(test_tags_encoded, pred_labels_encoded, average='weighted')
        print('Test f1 score', score)
        self.assertGreaterEqual(score, 0.15)
