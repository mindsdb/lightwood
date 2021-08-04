import unittest
import pandas as pd

from lightwood.api.high_level import json_ai_from_problem
from lightwood.api.types import ProblemDefinition


class TestModelSelection(unittest.TestCase):
    def get_models(self, df: pd.DataFrame, target: str, prob_kwargs: dict=None):
        prob_kwargs = {'target': target} if not prob_kwargs else {'target': target, **prob_kwargs}
        prdb = ProblemDefinition.from_dict(prob_kwargs)
        json_ai = json_ai_from_problem(df, prdb).to_dict()
        models = [model['module'] for model in json_ai['outputs'][target]['models']]
        return models

    def test_0_regression_task(self):
        df = pd.read_csv('tests/data/boston.csv')
        target = 'MEDV'
        expected_models = ['Neural', 'LightGBM', 'Regression']
        models = self.get_models(df, target)
        self.assertEqual(set(models), set(expected_models))

    def test_1_multiclass_task(self):
        df = pd.read_csv('tests/data/hdi.csv')
        target = 'Development Index'
        expected_models = ['Neural', 'LightGBM', 'Regression']
        models = self.get_models(df, target)
        self.assertEqual(set(models), set(expected_models))

    def test_2_unit_text_task(self):
        df = pd.read_csv('tests/data/tripadvisor_binary_sample.csv')
        target = 'Label'
        expected_models = ['Unit']
        models = self.get_models(df, target)
        self.assertEqual(set(models), set(expected_models))

    def test_3_complex_text_task(self):
        df = pd.read_csv('tests/data/wine_reviews_binary_sample.csv')
        target = 'label'
        expected_models = ['Neural', 'LightGBM', 'Regression']
        models = self.get_models(df, target)
        self.assertEqual(set(models), set(expected_models))

    def test_4_timeseries_t_plus_1(self):
        df = pd.read_csv('tests/data/arrivals.csv')
        target = 'Traffic'
        prob_kwargs = {
            'timeseries_settings': {
                'group_by': ['Country'],
                'nr_predictions': 1,
                'order_by': ['T'],
                'window': 5
            }
        }
        expected_models = ['Neural', 'LightGBM', 'Regression']
        models = self.get_models(df, target, prob_kwargs=prob_kwargs)
        self.assertEqual(set(models), set(expected_models))

    def test_5_timeseries_t_plus_n(self):
        df = pd.read_csv('tests/data/arrivals.csv')
        target = 'Traffic'
        prob_kwargs = {
            'timeseries_settings': {
                'group_by': ['Country'],
                'nr_predictions': 3,
                'order_by': ['T'],
                'window': 5
            }
        }
        expected_models = ['Neural', 'LightGBMArray', 'SkTime']
        models = self.get_models(df, target, prob_kwargs=prob_kwargs)
        self.assertEqual(set(models), set(expected_models))