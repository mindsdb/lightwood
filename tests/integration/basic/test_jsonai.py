import unittest
import pandas as pd
from lightwood.api.types import ProblemDefinition


class TestJsonAI(unittest.TestCase):
    def test_0_hidden_args_analysis(self):
        from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai

        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        jai = json_ai_from_problem(df, pdef)
        jai.analysis_blocks = [
            # args not needed (not even deps) because they are injected for default blocks
            {"module": "ICP"},
            {"module": "AccStats"},
            {"module": "ConfStats"},  # TODO: remove these three, they should always be enabled
            {"module": "GlobalFeatureImportance"}
        ]

        predictor = predictor_from_json_ai(jai)
        predictor.learn(df)
        self.assertTrue(len(predictor.analysis_blocks) == 4)
        self.assertTrue(all([0 <= colimp <= 1 for colimp in predictor.runtime_analyzer['column_importances'].values()]))
