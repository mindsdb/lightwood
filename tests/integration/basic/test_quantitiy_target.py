import unittest
import pandas as pd
from sklearn.metrics import r2_score

from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.DataFrame({
            'target': [f'{x}$' for x in range(1, 200)],
            'x': [x for x in range(1, 200)]
        })
        target = 'target'

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': target, 'time_aim': 200}))
        predictor.learn(df)
        predictions = predictor.predict(df)

        # sanity checks
        self.assertTrue(r2_score([float(x.rstrip('$')) for x in df[target]], predictions['prediction']) > 0.8)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
        self.assertTrue(all([p['lower'] <= p['prediction'] <= p['upper'] for _, p in predictions.iterrows()]))
