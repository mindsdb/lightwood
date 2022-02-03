import unittest
import pandas as pd
from sklearn.metrics import accuracy_score
from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
from lightwood.api.high_level import code_from_json_ai, predictor_from_code

from lightwood.mixer import QClassic


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        if QClassic is None:
            print('Skipping this test since the system for the quantum are not installed')
            return

        df = pd.read_csv('tests/data/hdi.csv')[:500]
        target = 'Development Index'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        json_ai = json_ai_from_problem(df, problem_definition=pdef)

        neural_args = json_ai.model['args']['submodels'][0]['args']
        neural_args["target_encoder"] = "$encoders[self.target]"
        neural_args["target"] = "$target"
        neural_args["dtype_dict"] = "$dtype_dict"
        neural_args["input_cols"] = "$input_cols"
        neural_args["timeseries_settings"] = "$problem_definition.timeseries_settings"
        neural_args["net"] = '"DefaultNet"'

        json_ai.model['args']['submodels'] = [{
            'module': 'lightwood.mixer.QClassic',
            'args': neural_args
        }]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df)
        return

        # sanity checks
        self.assertTrue(accuracy_score(df[target], predictions['prediction']) > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
