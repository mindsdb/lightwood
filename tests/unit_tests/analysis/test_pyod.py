import unittest
import pandas as pd
from lightwood.analysis import PyOD
from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
from lightwood.api.high_level import code_from_json_ai, predictor_from_code


class TestPyOD(unittest.TestCase):
    def test_0_pyod_analysis(self):
        if PyOD is None:
            print('Skipping this test since PyOD library is not installed')
            return

        dfs = [
            pd.read_csv('tests/data/concrete_strength.csv'),
            pd.read_csv('tests/data/hdi.csv'),
            pd.read_csv('tests/data/house_sales.csv'),
        ]
        dicts = [
            {'target': 'concrete_strength', 'time_aim': 40},
            {'target': 'Development Index', 'time_aim': 40},
            {'target': 'MA', 'time_aim': 40, 'timeseries_settings': {
                'group_by': ['bedrooms', 'type'],
                'horizon': 4,
                'order_by': 'saledate',
                'window': 8
            }},
        ]

        for df, d in zip(dfs, dicts):
            pdef = ProblemDefinition.from_dict(d)
            json_ai = json_ai_from_problem(df, problem_definition=pdef)

            json_ai.analysis_blocks = [{
                'module': 'lightwood.analysis.PyOD',
                'args': {}
            }]

            code = code_from_json_ai(json_ai)
            predictor = predictor_from_code(code)

            predictor.learn(df)
            predictions = predictor.predict(df.head())

            self.assertIn('pyod_explainer', predictor.runtime_analyzer)
            self.assertIn('pyod_anomaly', predictions.columns)
