from lightwood.api.json_ai import code_from_json_ai
import pandas as pd
from lightwood.api.types import DataAnalysis, JsonAI, ProblemDefinition
import importlib
import lightwood
from lightwood.api.predictor import PredictorInterface
import os
import tempfile


def analyze_dataset(df: pd.DataFrame, problem_definition_dict: dict = None) -> DataAnalysis:
    if problem_definition_dict is None:
        # Set a random target because some things expect that, won't matter for the analysis
        problem_definition_dict = {'target': str(df.columns[0])}
    problem_definition = ProblemDefinition.from_dict(problem_definition_dict)

    type_information = lightwood.data.infer_types(df, problem_definition.pct_invalid)
    statistical_analysis = lightwood.data.statistical_analysis(df, type_information, problem_definition)

    return DataAnalysis(
        type_information=type_information,
        statistical_analysis=statistical_analysis
    )


def code_from_problem(problem_definition: ProblemDefinition = None, data: pd.DataFrame = None, json_ai: JsonAI = None) -> str:
    if json_ai is None:
        type_information = lightwood.data.infer_types(data, problem_definition.pct_invalid)
        statistical_analysis = lightwood.data.statistical_analysis(data, type_information, problem_definition)
        json_ai = lightwood.generate_json_ai(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)

    predictor_code = code_from_json_ai(json_ai)
    # Runs OOM and takes forever if the code is very long

    return predictor_code


def predictor_from_code(code: str) -> PredictorInterface:
    # TODO: make this safe from code injection
    with tempfile.NamedTemporaryFile(suffix='.py') as temp:
        temp.write(code.encode('utf-8'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('a_temp_module', temp.name)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        predictor = temp_module.Predictor()
    return predictor


def predictor_from_problem(df: pd.DataFrame, problem_definition_dict: dict) -> PredictorInterface:
    predictor_class_str = code_from_problem(ProblemDefinition.from_dict(problem_definition_dict), df)

    return predictor_from_code(predictor_class_str)