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

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 40})
        json_ai = json_ai_from_problem(df, problem_definition=pdef)

        json_ai.analysis_blocks = [{
            'module': 'lightwood.analysis.ShapleyValues',
            'args': {}
        }]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df.head())

        self.assertIn('shap_explainer', predictor.runtime_analyzer)

        self.assertIn('shap_base_response', predictions.columns)
        self.assertIn('shap_final_response', predictions.columns)
        for input_col in df.columns:
            if input_col != target:
                self.assertIn(f'shap_contribution_{input_col}', predictions.columns)
