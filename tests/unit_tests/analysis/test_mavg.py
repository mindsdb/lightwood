import unittest
import pandas as pd

from lightwood.analysis import MAvg
from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
from lightwood.api.high_level import code_from_json_ai, predictor_from_code


class TestMAvg(unittest.TestCase):
    def test_0_mavg_analysis(self):
        if MAvg is None:
            print('Skipping MAvg test when values is empty')
            return

        df = pd.read_csv('tests/data/house_sales.csv')
        target = 'MA'

        pdef = ProblemDefinition.from_dict({
            'target': target,
            'timeseries_settings': {
                "order_by": "saledate",
                "group_by": ["type", "bedrooms"],
                "window": 8,
                "horizon": 4,
            }
        })
        json_ai = json_ai_from_problem(df, problem_definition=pdef)

        json_ai.analysis_blocks = [{
            'module': 'lightwood.analysis.MAvg',
            'args': {}
        }]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        preds = predictor.predict(df.head())

        # self.assertTrue(preds.columns)
