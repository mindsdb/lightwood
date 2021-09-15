import unittest
import pandas as pd
from sklearn.metrics import r2_score

from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/airline_sentiment.csv')[:50]
        target = 'airline_sentiment'

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': target, 'time_aim': 80}))
        predictor.learn(df)
        predictions = predictor.predict(df)

        # sanity checks
        self.assertTrue(r2_score(df[target], predictions['prediction']) > 0.8)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
        self.assertTrue(all([p['lower'] <= p['prediction'] <= p['upper'] for _, p in predictions.iterrows()]))
