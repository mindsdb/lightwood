import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai, JsonAI, code_from_json_ai, \
    predictor_from_code  # noqa

np.random.seed(42)


class TestBasic(unittest.TestCase):

    def get_submodels(self):
        submodels = [
            {
                'module': 'LightGBM',
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
        #df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        #target = 'concrete_strength'

        # generate data that mocks an observational skew by adding a linear selection to data
        data_size = 10000
        loc = 100.0
        target_data = np.random.normal(loc=loc, scale=1.0, size=data_size)
        epsilon = np.random.normal(loc=0.0, scale=.25, size=len(target_data))
        feature_data = target_data + epsilon
        df = pd.DataFrame({'feature': feature_data, 'target': target_data})

        hist, bin_edges = np.histogram(target_data, bins=10, density=False)
        fracs = np.linspace(.2, .8, len(hist))
        fracs = fracs / fracs.sum()
        skewed_arr_list = []
        for i in range(len(hist)):
            frac = fracs[i]
            low_edge = bin_edges[i]
            high_edge = bin_edges[i + 1]

            bin_array = target_data[target_data <= high_edge]
            bin_array = bin_array[bin_array >= low_edge]

            # select only a fraction fo the elements in this bin
            bin_array = bin_array[:int(len(bin_array) * frac)]

            skewed_arr_list.append(bin_array)

        skewed_arr = np.concatenate(skewed_arr_list)
        epsilon = np.random.normal(loc=0.0, scale=.25, size=len(skewed_arr))
        skewed_feat = skewed_arr + epsilon
        skew_df = pd.DataFrame({'feature': skewed_feat, 'target': skewed_arr})

        # generate data set weights to remove bias.
        hist, bin_edges = np.histogram(skew_df['target'].to_numpy(), bins=10, density=False)
        hist = 1 - hist / hist.sum()
        target_weights = {bin_edge: bin_frac for bin_edge, bin_frac in zip(bin_edges, hist)}

        pdef = ProblemDefinition.from_dict({'target': 'target', 'target_weights': target_weights, 'time_aim': 80})
        print(pdef.to_dict())
        jai = json_ai_from_problem(skew_df, pdef)

        jai.model['args']['submodels'] = self.get_submodels()
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(skew_df)
        output_df = predictor.predict(df)

        self.assertTrue(np.all(np.isclose(output_df['prediction'].mean(), loc, atol=0., rtol=.1)))




"""    def test_1_binary(self):
        df = pd.read_csv('tests/data/ionosphere.csv')[:100]
        target = 'target'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 20, 'unbias_target': False})
        jai = json_ai_from_problem(df, pdef)

        jai.model['args']['submodels'] = self.get_submodels()
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df)

        acc = balanced_accuracy_score(df[target], predictions['prediction'])
        self.assertTrue(acc > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))"""
