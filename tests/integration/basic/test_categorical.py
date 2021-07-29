import unittest
import numpy as np
import pandas as pd
from lightwood.api.types import ProblemDefinition
np.random.seed(42)


class TestBasic(unittest.TestCase):
    def test_0_categorical(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/adult.csv')[:500]
        target = 'income'

        mask = np.random.rand(len(df)) < 0.8
        train = df[mask]
        test = df[~mask]

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': 'income', 'time_aim': 100}))
        predictor.learn(train)

        if hasattr(predictor, 'ensemble'):
            for i, model in enumerate(predictor.ensemble.models):

                predictor.ensemble.best_index = i
                predictions = predictor.predict(test)
                assert 'truth' in predictions.columns
                assert 'prediction' in predictions.columns
                assert 'confidence' in predictions.columns

                predictions = predictor.predict_proba(test)

                for label in df[target].unique():
                    assert f'__mdb_proba_{label}' in predictions.columns