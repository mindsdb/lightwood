import unittest
import pandas as pd
from sklearn.metrics import accuracy_score
from tests.utils.timing import train_and_check_time_aim
from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    # Interesting: has coordinates as inputs
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/airline_sentiment.csv')[:500]
        target = 'airline_sentiment'

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': target, 'time_aim': 80}))
        train_and_check_time_aim(predictor, df)
        predictions = predictor.predict(df)

        # sanity checks
        self.assertTrue(accuracy_score(df[target], predictions['prediction']) > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
