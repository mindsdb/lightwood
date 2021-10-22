from lightwood.api.dtype import dtype
import unittest
import pandas as pd
from sklearn.metrics import r2_score

from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        # Mess with the names to also test if lightwood can deal /w weird names
        df = df.rename(columns={df.columns[1]: f'\'{df.columns[1]}\''})
        df = df.rename(columns={df.columns[2]: f'\'{df.columns[2]}}}'})
        df = df.rename(columns={df.columns[3]: f'{{{df.columns[3]}\"'})
        target = 'concrete_strength'

        # Make this a quantity
        df[target] = [f'{x}$' for x in df[target]]
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 200})

        predictor = predictor_from_problem(df, pdef)
        predictor.learn(df)

        assert predictor.model_analysis.dtypes[target] == dtype.quantity

        predictions = predictor.predict(df)

        # sanity checks
        self.assertTrue(r2_score([float(x.rstrip('$')) for x in df[target]], predictions['prediction']) > 0.8)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
        self.assertTrue(all([p['lower'] <= p['prediction'] <= p['upper'] for _, p in predictions.iterrows()]))

        # check customizable ICP fixed confidence param
        fixed_conf = 0.8
        fixed_predictions = predictor.predict(df, {'fixed_confidence': fixed_conf})

        assert all([v == fixed_conf for v in fixed_predictions['confidence'].values])
