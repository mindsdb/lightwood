import unittest
import pandas as pd
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai, JsonAI, code_from_json_ai, predictor_from_code  # noqa


class TestJsonAI(unittest.TestCase):
    def test_0_hidden_args_analysis(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'
        pdef = ProblemDefinition.from_dict({
            'target': target,
            'time_aim': 80,
            'dtype_dict': {'age': 'categorical'}  # tests overriding inferred types
        })
        jai = json_ai_from_problem(df, pdef)
        jai.analysis_blocks = [
            # args not needed (not even deps), they should be injected for default blocks
            {"module": "PermutationFeatureImportance"}
        ]

        predictor = predictor_from_json_ai(jai)
        predictor.learn(df)
        self.assertTrue(len(predictor.analysis_blocks) == 4)

    def test_1_incorrect_chain(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        jai = json_ai_from_problem(df, pdef)
        jai.analysis_blocks = [{'module': 'TempScaler', 'args': {'deps': ['Some other block']}}]
        with self.assertRaises(Exception) as context:
            predictor_from_json_ai(jai)
        self.assertTrue('not found but necessary for block' in str(context.exception))

    def test_2_tempscale_analysis(self):
        # Create base json ai
        df = pd.read_csv('tests/data/hdi.csv').iloc[0:100]
        pdef = ProblemDefinition.from_dict({'target': 'Development Index',
                                            'time_aim': 20,
                                            'use_default_analysis': False
                                            })
        json_ai = json_ai_from_problem(df, pdef)

        # modify it
        json_ai_dump = json_ai.to_dict()
        json_ai_dump['analysis_blocks'] = [{'module': 'TempScaler'}]
        json_ai = JsonAI.from_dict(json_ai_dump)

        # create a predictor from it
        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        row_insights = predictor.predict(df)
        assert 'confidence' in row_insights.columns

    def test_3_featurize_hdf(self):
        import os
        import tempfile

        tempdir = tempfile.mkdtemp()
        os.environ['LIGHTWOOD_DEV_SAVE_TO'] = os.path.join(tempdir, 'predictor.py')

        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 10})
        jai = json_ai_from_problem(df, pdef)
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        # training run #1, generates hdf5 file
        predictor.learn(df)
        predictor.predict(df)
        assert os.path.exists(tempdir + '/lightwood_features.h5')

        # training run #2, hdf5 file is used
        predictor.learn(df)
        predictor.predict(df)

        # training run #3, different predictor, hdf5 is rebuilt
        target = 'age'
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 10})
        jai = json_ai_from_problem(df, pdef)
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        predictor.predict(df)
