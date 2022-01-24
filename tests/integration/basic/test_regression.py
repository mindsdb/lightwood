from lightwood.api.dtype import dtype
import unittest
import pandas as pd
from sklearn.metrics import r2_score
from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai

        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        # Mess with the names to also test if lightwood can deal /w weird names
        df = df.rename(columns={df.columns[1]: f'\'{df.columns[1]}\''})
        df = df.rename(columns={df.columns[2]: f'\'{df.columns[2]}}}'})
        df = df.rename(columns={df.columns[3]: f'{{{df.columns[3]}\"'})
        target = 'concrete_strength'

        # Make this a quantity
        df[target] = [f'{x}$' for x in df[target]]
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        jai = json_ai_from_problem(df, pdef)
        jai.analysis_blocks = [{
            "module": "ICP",
            "args": {
                "fixed_significance": None,
                "confidence_normalizer": True,  # explicitly test the ICP normalizer in an integration test
                "positive_domain": "$statistical_analysis.positive_domain",
            }},
            {
                "module": "AccStats",
                "args": {"deps": ["ICP"]}
        },
            {
                "module": "ConfStats",
                "args": {"deps": ["ICP"]}
        }]

        predictor = predictor_from_json_ai(jai)
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

        # test empty dataframe handling
        with self.assertRaises(Exception) as ctx:
            predictor.predict(pd.DataFrame())
        self.assertTrue('Empty input' in str(ctx.exception))
