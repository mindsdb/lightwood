import unittest
import pandas as pd

from lightwood.api.high_level import json_ai_from_problem
from lightwood.api.types import ProblemDefinition


class TestMixerSelection(unittest.TestCase):
    def get_mixers(self, df: pd.DataFrame, target: str, prob_kwargs: dict = None):
        prob_kwargs = {'target': target, 'time_aim': 15} if not prob_kwargs else {'target': target, **prob_kwargs}
        prdb = ProblemDefinition.from_dict(prob_kwargs)
        json_ai = json_ai_from_problem(df, prdb).to_dict()
        mixers = [mixer['module'] for mixer in json_ai['model']['args']['submodels']]
        return mixers

    def test_0_regression_task(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')
        target = 'concrete_strength'
        expected_mixers = ['Neural', 'LightGBM', 'Regression']
        mixers = self.get_mixers(df, target)
        self.assertEqual(set(mixers), set(expected_mixers))

    def test_1_multiclass_task(self):
        df = pd.read_csv('tests/data/hdi.csv')
        target = 'Development Index'
        expected_mixers = ['Neural', 'LightGBM', 'Regression']
        mixers = self.get_mixers(df, target)
        self.assertEqual(set(mixers), set(expected_mixers))

    def test_2_unit_text_task(self):
        df = pd.read_csv('tests/data/tripadvisor_binary_sample.csv')
        target = 'Label'
        expected_mixers = ['Unit']
        mixers = self.get_mixers(df, target)
        self.assertEqual(set(mixers), set(expected_mixers))

    def test_3_complex_text_task(self):
        df = pd.read_csv('tests/data/wine_reviews_binary_sample.csv')
        target = 'label'
        expected_mixers = ['Neural', 'LightGBM', 'Regression']
        mixers = self.get_mixers(df, target)
        self.assertEqual(set(mixers), set(expected_mixers))

    def test_4_timeseries_t_plus_1(self):
        df = pd.read_csv('tests/data/arrivals.csv')
        target = 'Traffic'
        prob_kwargs = {
            'timeseries_settings': {
                'time_aim': 15,
                'group_by': ['Country'],
                'horizon': 1,
                'order_by': 'T',
                'window': 5
            }
        }
        expected_mixers = ['NeuralTs', 'LightGBM', 'Regression']
        mixers = self.get_mixers(df, target, prob_kwargs=prob_kwargs)
        self.assertEqual(set(mixers), set(expected_mixers))

    def test_5_timeseries_t_plus_n(self):
        df = pd.read_csv('tests/data/arrivals.csv')
        target = 'Traffic'
        prob_kwargs = {
            'time_aim': 15,
            'timeseries_settings': {
                'group_by': ['Country'],
                'horizon': 3,
                'order_by': 'T',
                'window': 5
            }
        }
        expected_mixers = ['NeuralTs', 'LightGBMArray', 'SkTime']
        mixers = self.get_mixers(df, target, prob_kwargs=prob_kwargs)
        self.assertEqual(set(mixers), set(expected_mixers))
