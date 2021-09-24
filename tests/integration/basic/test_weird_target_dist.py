import unittest
import pandas as pd
from lightwood.api.types import ProblemDefinition
from lightwood import dtype


class TestBasic(unittest.TestCase):
    def test_0_unkown_cateogires_in_test(self):
        """This test is meant to check if the expaliner can handle a  validation set\
            with values that it has never seen as part of the training set.
            In order to achieve this, we generate a dataset with a target that contains 600\
            values, out of which 100 are distinct. It's basically certain at least 1 of these\
            distinct value will be in train and at least one in validate (more likely 90/10 split)
            So if training and predicting goes well, this validates that explainer works with\
            new values in the validation set *and* with values in train that never occur in validation.
            Note: This doesn't check how good the confidence is, it'll likely be very bad, we just care\
            that it doesn't crash.
        """
        from lightwood.api.high_level import predictor_from_problem

        # The target will be cateogircal and there will be a bunch of values
        # in all datasets (train/dev/validation) that were not present in the others
        df = pd.DataFrame({
            'target': [1 for _ in range(500)] + [f'{i}cat' for i in range(100)],
            'y': [i for i in range(600)]
        })
        target = 'target'

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict(
                                           {'target': target, 'time_aim': 60, 'unbias_target': True}))
        predictor.learn(df)
        assert predictor.model_analysis.dtypes['target'] == dtype.categorical
        predictor.predict(df)
