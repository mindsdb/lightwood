import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai, JsonAI, code_from_json_ai, predictor_from_code  # noqa


np.random.seed(42)


class TestBasic(unittest.TestCase):

    def get_submodels(self):
        submodels = [
            {
                'module': 'RandomForest',
                'args': {
                    'stop_after': '$problem_definition.seconds_per_mixer',
                    'fit_on_dev': True,
                    'target': '$target',
                    'dtype_dict': '$dtype_dict',
                    'target_encoder': '$encoders[self.target]',
                    'use_optuna': True
                }
            },
        ]
        return submodels

    def test_0_regression(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        jai = json_ai_from_problem(df, pdef)

        jai.model['args']['submodels'] = self.get_submodels()
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictor.predict(df)

    def test_1_binary(self):
        df = pd.read_csv('tests/data/ionosphere.csv')[:100]
        target = 'target'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 20, 'unbias_target': False})
        jai = json_ai_from_problem(df, pdef)

        jai.model['args']['submodels'] = [
            {
                'module': 'RandomForest',
                'args': {'stop_after': '$problem_definition.seconds_per_mixer', 'fit_on_dev': True}}
        ]
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df)

        acc = balanced_accuracy_score(df[target], predictions['prediction'])
        self.assertTrue(acc > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))
