import unittest
import pandas as pd
from sklearn.metrics import accuracy_score

from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
from lightwood.api.high_level import code_from_json_ai, predictor_from_code


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):

        df = pd.read_csv('tests/data/hdi.csv')[:500]
        target = 'Development Index'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 10})
        json_ai = json_ai_from_problem(df, problem_definition=pdef)

        json_ai.analysis_blocks = [{
            'module': 'lightwood.analysis.ShapleyValues',
            'args': {}
        }]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df)

        self.assertTrue(accuracy_score(df[target], predictions['prediction'].astype(int)) > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))

