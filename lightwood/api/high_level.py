import os
import pandas as pd
from lightwood.api.types import DataAnalysis, JsonAI, ProblemDefinition
import lightwood
from lightwood.api.predictor import PredictorInterface
from lightwood.api.json_ai import generate_json_ai
import tempfile
from lightwood.api.json_ai import code_from_json_ai as _code_from_json_ai
import importlib.util
import sys
import random
import string
import dill


def _module_from_code(code, module_name):
    dev_file = os.environ.get('LIGHTWOOD_DEV_SAVE_TO', None)
    if dev_file is not None:
        fp = open(dev_file, 'wb')
    else:
        fp = tempfile.NamedTemporaryFile(suffix='.py')

    fp.write(code.encode('utf-8'))
    spec = importlib.util.spec_from_file_location(module_name, fp.name)
    temp_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = temp_module
    spec.loader.exec_module(temp_module)

    fp.close()
    return temp_module


def code_from_json_ai(json_ai: JsonAI) -> str:
    return _code_from_json_ai(json_ai)


def analyze_dataset(df: pd.DataFrame) -> DataAnalysis:
    problem_definition = ProblemDefinition.from_dict({'target': str(df.columns[0])})

    type_information = lightwood.data.infer_types(df, problem_definition.pct_invalid)
    statistical_analysis = lightwood.data.statistical_analysis(df, type_information, problem_definition)

    return DataAnalysis(
        type_information=type_information,
        statistical_analysis=statistical_analysis
    )


def json_ai_from_problem(df: pd.DataFrame, problem_definition: ProblemDefinition) -> JsonAI:
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    type_information = lightwood.data.infer_types(df, problem_definition.pct_invalid)
    statistical_analysis = lightwood.data.statistical_analysis(df, type_information, problem_definition)
    json_ai = generate_json_ai(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)

    return json_ai


def code_from_problem(df: pd.DataFrame, problem_definition: ProblemDefinition) -> str:
    json_ai = json_ai_from_problem(df, problem_definition)
    predictor_code = code_from_json_ai(json_ai)
    return predictor_code


def predictor_from_code(code: str) -> PredictorInterface:
    module_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    predictor = _module_from_code(code, module_name).Predictor()
    return predictor


def predictor_from_state(state_file: str, code: str = None) -> PredictorInterface:
    with open(state_file, 'rb') as fp:
        try:
            module_name = None
            predictor = dill.load(fp)
        except Exception as e:
            module_name = str(e).lstrip("No module named '").split("'")[0]
            if code is None:
                raise Exception('You need to provide the code if trying to load a predictor from outside the scope/script it was created in!')
        
        if module_name is not None:
            _module_from_code(code, module_name)
            predictor = dill.load(fp)

    return predictor



def predictor_from_problem(df: pd.DataFrame, problem_definition: ProblemDefinition) -> PredictorInterface:
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    predictor_class_str = code_from_problem(df, problem_definition)
    return predictor_from_code(predictor_class_str)
