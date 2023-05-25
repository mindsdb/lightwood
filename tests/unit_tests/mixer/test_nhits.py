import unittest
import numpy as np
import pandas as pd
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai, JsonAI, code_from_json_ai, predictor_from_code  # noqa


np.random.seed(42)


class TestBasic(unittest.TestCase):
    def get_submodels(self):
        submodels = [
            {
                'module': 'NHitsMixer',
                'args': {
                    'train_args': {
                        'trainer_args': {
                            'max_epochs': 10,
                            'conf_level': [90, 95],
                        },
                    }
                }
            },
        ]
        return submodels

    def test_0_regression(self):
        from lightwood.mixer import NHitsMixer
        if NHitsMixer is not None:
            df = pd.read_csv('tests/data/arrivals.csv')
            target = 'Traffic'

            pdef = ProblemDefinition.from_dict({'target': target, 'timeseries_settings': {'order_by': 'T',
                                                                                          'group_by': ['Country'],
                                                                                          'window': 4,
                                                                                          'horizon': 4}})
            jai = json_ai_from_problem(df, pdef)
            jai.model['args']['submodels'] = self.get_submodels()
            code = code_from_json_ai(jai)
            predictor = predictor_from_code(code)

            predictor.learn(df)
            predictor.predict(df, args={'fixed_confidence': 0.9})
