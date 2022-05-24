import os
from types import ModuleType
from typing import Union
import dill
import pandas as pd
from lightwood.api.types import DataAnalysis, JsonAI, ProblemDefinition
from lightwood.data import statistical_analysis
from lightwood.data import infer_types
from lightwood.api.predictor import PredictorInterface
from lightwood.api.json_ai import generate_json_ai
import tempfile
from lightwood.api.json_ai import code_from_json_ai as _code_from_json_ai
import importlib.util
import sys
import random
import string
import gc
import time
from lightwood.helpers.log import log
from shutil import copyfile


def load_custom_module(file_path: str):
    from lightwood import __version__ as lightwood_version
    modules_dir = os.path.join(os.path.expanduser('~/lightwood_modules'), lightwood_version.replace('.', '_'))
    custom_module_path = os.path.join(modules_dir, os.path.split(file_path)[-1])
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir)
    if os.path.exists(custom_module_path):
        os.remove(custom_module_path)

    copyfile(file_path, custom_module_path)


def predictor_from_problem(df: pd.DataFrame, problem_definition: Union[ProblemDefinition, dict]) -> PredictorInterface:
    """
    Creates a ready-to-train ``Predictor`` object from some raw data and a ``ProblemDefinition``. Do not use this if you want to edit the JsonAI first. Usually you'd want to next train this predictor by calling the ``learn`` method on the same dataframe used to create it.

    :param df: The raw data
    :param problem_definition: The manual specifications for your predictive problem

    :returns: A lightwood ``Predictor`` object
    """ # noqa
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    if problem_definition.ignore_features:
        log.info(f'Dropping features: {problem_definition.ignore_features}')
        df = df.drop(columns=problem_definition.ignore_features)

    predictor_class_str = code_from_problem(df, problem_definition)
    return predictor_from_code(predictor_class_str)


def json_ai_from_problem(df: pd.DataFrame, problem_definition: Union[ProblemDefinition, dict]) -> JsonAI:
    """
    Creates a JsonAI from your raw data and problem definition. Usually you would use this when you want to subsequently edit the JsonAI, the easiest way to do this is to unload it to a dictionary via `to_dict`, modify it, and then create a new object from it using `lightwood.JsonAI.from_dict`. It's usually better to generate the JsonAI using this function rather than writing it from scratch.

    :param df: The raw data
    :param problem_definition: The manual specifications for your predictive problem

    :returns: A ``JsonAI`` object generated based on your data and problem specifications
    """ # noqa
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    started = time.time()

    if problem_definition.ignore_features:
        log.info(f'Dropping features: {problem_definition.ignore_features}')
        df = df.drop(columns=problem_definition.ignore_features)

    type_information = infer_types(df, problem_definition.pct_invalid)
    stats = statistical_analysis(
        df, type_information.dtypes, type_information.identifiers, problem_definition)

    duration = time.time() - started
    if problem_definition.time_aim is not None:
        problem_definition.time_aim -= duration
        if problem_definition.time_aim < 10:
            problem_definition.time_aim = 10

    # Assume that the stuff besides encoder and mixers takes about as long as analyzing did... bad, but let's see
    if problem_definition.expected_additional_time is None:
        problem_definition.expected_additional_time = duration
    json_ai = generate_json_ai(
        type_information=type_information, statistical_analysis=stats,
        problem_definition=problem_definition)

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    """
    Autogenerates custom code based on the details you specified inside your JsonAI.

    :param json_ai: A ``JsonAI`` object

    :returns: Code (text) generate based on the ``JsonAI`` you created
    """
    return _code_from_json_ai(json_ai)


def predictor_from_code(code: str) -> PredictorInterface:
    """
    :param code: The ``Predictor``'s code in text form

    :returns: A lightwood ``Predictor`` object
    """
    module_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    module_name += str(time.time()).replace('.', '')
    predictor = _module_from_code(code, module_name).Predictor()
    return predictor


def analyze_dataset(df: pd.DataFrame) -> DataAnalysis:
    """
    You can use this to understand and visualize the data, it's not a part of the pipeline one would use for creating and training predictive models.

    :param df: The raw data

    :returns: An object containing insights about the data (specifically the type information and statistical analysis)
    """ # noqa

    problem_definition = ProblemDefinition.from_dict({'target': str(df.columns[0])})

    type_information = infer_types(df, problem_definition.pct_invalid)
    stats = statistical_analysis(
        df, type_information.dtypes, type_information.identifiers, problem_definition)

    return DataAnalysis(
        type_information=type_information,
        statistical_analysis=stats
    )


def code_from_problem(df: pd.DataFrame, problem_definition: Union[ProblemDefinition, dict]) -> str:
    """
    :param df: The raw data
    :param problem_definition: The manual specifications for your predictive problem

    :returns: The text code generated based on your data and problem specifications
    """
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    if problem_definition.ignore_features:
        log.info(f'Dropping features: {problem_definition.ignore_features}')
        df = df.drop(columns=problem_definition.ignore_features)

    json_ai = json_ai_from_problem(df, problem_definition)
    predictor_code = code_from_json_ai(json_ai)
    return predictor_code


def predictor_from_state(state_file: str, code: str = None) -> PredictorInterface:
    """
    :param state_file: The file containing the pickle resulting from calling ``save`` on a ``Predictor`` object
    :param code: The ``Predictor``'s code in text form

    :returns: A lightwood ``Predictor`` object
    """
    try:
        module_name = None
        with open(state_file, 'rb') as fp:
            predictor = dill.load(fp)
    except Exception as e:
        module_name = str(e).lstrip("No module named '").split("'")[0]
        if code is None:
            raise Exception(
                'Provide code when loading a predictor from outside the scope/script it was created in!')

    if module_name is not None:
        try:
            del sys.modules[module_name]
        except Exception:
            pass
        gc.collect()
        _module_from_code(code, module_name)
        with open(state_file, 'rb') as fp:
            predictor = dill.load(fp)

    return predictor


def _module_from_code(code: str, module_name: str) -> ModuleType:
    """
    Create a python module (containing the generated ``Predictor`` class) from the code. This is both a python object and an associated temporary file on your filesystem

    :param code: The ``Predictor``'s code in text form
    :param module_name: The name of the newly created module

    :returns: A python module object
    """ # noqa
    dirname = tempfile.gettempdir()
    filename = os.urandom(24).hex() + str(time.time()).replace('.', '') + '.py'
    path = os.path.join(dirname, filename)
    if 'LIGHTWOOD_DEV_SAVE_TO' in os.environ:
        path = os.environ['LIGHTWOOD_DEV_SAVE_TO']

    with open(path, 'wb') as fp:
        fp.write(code.encode('utf-8'))
        spec = importlib.util.spec_from_file_location(module_name, fp.name)
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = temp_module
        spec.loader.exec_module(temp_module)

    return temp_module


def predictor_from_json_ai(json_ai: JsonAI) -> PredictorInterface:
    """
    Creates a ready-to-train ``Predictor`` object based on the details you specified inside your JsonAI.

    :param json_ai: A ``JsonAI`` object

    :returns: A lightwood ``Predictor`` object
    """  # noqa
    code = code_from_json_ai(json_ai)
    predictor = predictor_from_code(code)
    return predictor
