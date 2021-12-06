from lightwood.api.high_level import json_ai_from_problem, code_from_json_ai, predictor_from_code, load_custom_module
from lightwood.api.types import JsonAI, ProblemDefinition
import unittest
import os
import pandas as pd


test_err_message = 'This ! Is ! A ! Testing ! Error !'
mdir = os.path.expanduser('~/lightwood_modules')


def create_custom_module(module_name, module_code):
    with open(module_name, 'w') as fp:
        fp.write(module_code)

    load_custom_module(module_name)
    os.remove(module_name)


class TestBasic(unittest.TestCase):
    def test_0_add_throwing_cleaner(self):
        module_code = """
import pandas as pd

def throwing_cleaner(data: pd.DataFrame, err_msg: str):
    assert isinstance(data, pd.DataFrame)
    raise Exception(err_msg)
"""
        create_custom_module('custom_cleaners.py', module_code)

        # Create base json ai
        df = pd.read_csv('tests/data/hdi.csv').iloc[0:400]
        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({'target': 'Development Index', 'time_aim': 20}))

        # modify it
        json_ai_dump = json_ai.to_dict()
        json_ai_dump['cleaner'] = {
            'module': 'custom_cleaners.throwing_cleaner',
            'args': {
                'err_msg': f'"{test_err_message}"'
            }
        }

        json_ai = JsonAI.from_dict(json_ai_dump)

        # create a predictor from it
        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        try:
            predictor.learn(df)
        except Exception as e:
            assert str(e) == test_err_message
            return

        raise Exception('Predictor did not contain modified function!')

    def test_1_add_analyzer_block(self):

        mname = 'custom_analyzers'
        cname = 'ExampleAnalysis'
        module_code = f"""
from lightwood.analysis.base import BaseAnalysisBlock

class {cname}(BaseAnalysisBlock):
    def __init__(self):
        super().__init__(deps=None)

    def analyze(self, info, **kwargs):
        info['test'] = 'test'
        return info

    def explain(self, row_insights, global_insights, **kwargs):
        row_insights['test'] = 'test'
        return row_insights, global_insights
"""
        create_custom_module(f'{mname}.py', module_code)

        # Create base json ai
        df = pd.read_csv('tests/data/hdi.csv').iloc[0:400]
        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({'target': 'Development Index', 'time_aim': 20}))

        # modify it
        json_ai_dump = json_ai.to_dict()
        json_ai_dump['analysis_blocks'] = [{
            'module': f'{mname}.{cname}',
            'args': {}
        }]

        json_ai = JsonAI.from_dict(json_ai_dump)

        # create a predictor from it
        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        row_insights = predictor.predict(df)

        assert predictor.runtime_analyzer['test'] == 'test'
        assert row_insights['test'].iloc[0] == 'test'
