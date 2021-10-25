import unittest
import pandas as pd
from lightwood.analysis import ShapleyValues

from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
from lightwood.api.high_level import code_from_json_ai, predictor_from_code


class TestBasic(unittest.TestCase):
    def test_0_shap_analysis(self):
        if ShapleyValues is None:
            print('Skipping this test since the Shapley values library is not installed')
            return

        df = pd.read_csv('tests/data/hdi.csv')
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
        predictor.predict(df)

        assert 'shap_explainer' in predictor.runtime_analyzer

