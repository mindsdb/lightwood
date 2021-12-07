from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import predictor_from_problem
import pandas as pd
import unittest


class TestText(unittest.TestCase):
    def test_0_train_and_predict_bypass(self):
        df = pd.read_csv('tests/data/tripadvisor_binary_sample.csv')[:100]
        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({
            'target': 'Label', 'time_aim': 40
        }))
        # Test with a short time aim
        predictor.learn(df)
        predictions = predictor.predict(df)
        for x in predictions['prediction']:
            assert x is not None

    # Comment back in when we can reduce the time for these more, for now this is impossible
    # Datasource building for LGBM /w text just takes too long
    def test_1_train_and_predict_model(self):
        df = pd.read_csv('tests/data/wine_reviews_binary_sample.csv')[:100]
        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({
            'target': 'label', 'time_aim': 40
        }))
        # Test with a longer time aim
        predictor.learn(df)
        predictions = predictor.predict(df)
        for x in predictions['prediction']:
            assert x is not None
