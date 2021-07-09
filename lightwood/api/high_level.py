import pandas as pd
from lightwood.api.types import DataAnalysis, JsonAI, ProblemDefinition
import lightwood
from lightwood.api.predictor import PredictorInterface
from lightwood.api.json_ai import generate_json_ai
import tempfile
from lightwood.api.json_ai import code_from_json_ai as _code_from_json_ai


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


def predictor_from_code(code: str, return_class: bool = False) -> PredictorInterface:
    # TODO: make this safe from code injection
    with tempfile.NamedTemporaryFile(suffix='.py') as temp:
        temp.write(code.encode('utf-8'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('a_temp_module', temp.name)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)
        if return_class:
            return temp_module.Predictor
        predictor = temp_module.Predictor()
    return predictor


def predictor_from_problem(df: pd.DataFrame, problem_definition: ProblemDefinition, return_class: bool = False) -> PredictorInterface:
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    predictor_class_str = code_from_problem(df, problem_definition)
    return predictor_from_code(predictor_class_str, return_class)
