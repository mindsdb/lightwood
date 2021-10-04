from lightwood.api.high_level import json_ai_from_problem, code_from_json_ai, predictor_from_code
from lightwood.api.types import JsonAI, ProblemDefinition
import unittest
from mindsdb_datasources import FileDS
import os
import shutil


test_err_message = 'This ! Is ! A ! Testing ! Error !'
mdir = os.path.expanduser('~/lightwood_modules')


def create_custom_module(mpath, mcode):
    try:
        shutil.rmtree(mpath)
    except Exception:
        pass

    try:
        os.mkdir(mdir)
    except Exception:
        pass

    with open(mpath, 'w') as fp:
        fp.write(mcode)


class TestBasic(unittest.TestCase):
    def test_0_add_throwing_cleaner(self):
        module_code = """
import pandas as pd

def throwing_cleaner(data: pd.DataFrame, err_msg: str):
    assert isinstance(data, pd.DataFrame)
    raise Exception(err_msg)
"""
        create_custom_module(os.path.join(mdir, 'custom_cleaners.py'), module_code)

        # Create base json ai
        df = FileDS('tests/data/hdi.csv').df.iloc[0:400]
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
        create_custom_module(os.path.join(mdir, f'{mname}.py'), module_code)

        # Create base json ai
        df = FileDS('tests/data/hdi.csv').df.iloc[0:400]
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
