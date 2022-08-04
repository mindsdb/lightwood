import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import predictor_from_problem  # noqa
np.random.seed(42)


class TestBasic(unittest.TestCase):
    def setup_predictor(self, df, target):
        mask = np.random.rand(len(df)) < 0.8
        train = df[mask]
        test = df[~mask]

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict(
            {'target': target, 'time_aim': 20, 'unbias_target': False}))

        predictor.learn(train)

        if hasattr(predictor, 'ensemble'):
            for i, mixer in enumerate(predictor.ensemble.mixers):

                predictor.ensemble.best_index = i
                predictions = predictor.predict(test)
                assert 'prediction' in predictions.columns
                assert 'confidence' in predictions.columns

                predictions = predictor.predict(test, args={'predict_proba': True})

                for label in df[target].unique():
                    assert f'__mdb_proba_{label}' in predictions.columns
        return predictor

    def test_0_binary(self):
        df = pd.read_csv('tests/data/ionosphere.csv')[:100]
        target = 'target'
        predictor = self.setup_predictor(df, target)
        predictions = predictor.predict(df)
        acc = balanced_accuracy_score(df[target], predictions['prediction'])
        self.assertTrue(acc > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))

    def test_1_categorical(self):
        df = pd.read_csv('tests/data/hdi.csv')
        target = 'Development Index'
        predictor = self.setup_predictor(df, target)
        predictions = predictor.predict(df)

        self.assertTrue(balanced_accuracy_score(df[target].astype(int), predictions['prediction'].astype(int)) > 0.9)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))

        # test predict all mixers with some data
        predictions = predictor.predict(df[:10], args={'all_mixers': True})
        assert '__mdb_mixer_Neural' in predictions.columns

        # predict single sample
        predictor.predict(df.iloc[[0]])

    def test_2_binary_no_analysis(self):
        df = pd.read_csv('tests/data/ionosphere.csv')[:100]
        mask = np.random.rand(len(df)) < 0.8
        train = df[mask]
        test = df[~mask]
        predictor = predictor_from_problem(df, ProblemDefinition.from_dict(
            {'target': 'target', 'time_aim': 20, 'use_default_analysis': False}))
        predictor.learn(train)
        predictions = predictor.predict(test)
        self.assertTrue(balanced_accuracy_score(test['target'], predictions['prediction']) > 0.5)
        self.assertTrue('confidence' not in predictions.columns)

