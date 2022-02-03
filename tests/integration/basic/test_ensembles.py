import unittest
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from lightwood.api.high_level import code_from_json_ai, json_ai_from_problem, predictor_from_code
from lightwood.api.types import ProblemDefinition


class TestBasic(unittest.TestCase):
    def test_0_mean_ensemble(self):

        df = pd.read_csv('tests/data/concrete_strength.csv')

        target = 'concrete_strength'

        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({
            'target': target,
            'time_aim': 80
        }))

        json_ai.model = {
            'module': 'MeanEnsemble',
            "args": {
                'dtype_dict': '$dtype_dict',
                'submodels': json_ai.model['args']['submodels']
            }
        }

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        predictions = predictor.predict(df)

        self.assertTrue(r2_score(df[target], predictions['prediction']) > 0.5)

    def test_1_mode_ensemble(self):

        df = pd.read_csv('tests/data/hdi.csv')

        target = 'Development Index'

        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({
            'target': target,
            'time_aim': 5
        }))

        json_ai.model = {
            'module': 'ModeEnsemble',
            "args": {
                'dtype_dict': '$dtype_dict',
                "args": "$pred_args",
                "accuracy_functions": "$accuracy_functions",
                'submodels': json_ai.model['args']['submodels']
            }
        }

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        predictions = predictor.predict(df)

        self.assertTrue(accuracy_score(df[target].astype(int), predictions['prediction'].astype(int)) > 0.5)

    def test_2_weighted_mean_ensemble(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')

        target = 'concrete_strength'

        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({
            'target': target,
            'time_aim': 80
        }))

        json_ai.model = {
            'module': 'WeightedMeanEnsemble',
            "args": {
                'dtype_dict': '$dtype_dict',
                "args": "$pred_args",
                "accuracy_functions": "$accuracy_functions",
                'submodels': json_ai.model['args']['submodels']
            }
        }

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        predictions = predictor.predict(df)

        self.assertTrue(r2_score(df[target], predictions['prediction']) > 0.5)
