from numpy import DataSource
from lightwood.api.types import DataAnalysis, ProblemDefinition
import importlib
from lightwood.api.generate_predictor import generate_predictor
import lightwood


def make_predictor(datasource: DataSource, problem_definition_dict: dict):
    predictor_class_str = generate_predictor(ProblemDefinition.from_dict(problem_definition_dict), datasource.df)

    with open('dynamic_predictor.py', 'w') as fp:
        fp.write(predictor_class_str)

    predictor_class = importlib.import_module('dynamic_predictor').Predictor
    print('Class was evaluated successfully')

    predictor = predictor_class()
    print('Class initialized successfully')

    return predictor


def analyze_dataset(datasource: DataSource, problem_definition_dict: dict = None):
    if problem_definition_dict is None:
        problem_definition_dict = {}
    problem_definition = ProblemDefinition.from_dict(problem_definition_dict)

    df = datasource.df
    type_information = lightwood.data.infer_types(df, problem_definition.pct_invalid)
    statistical_analysis = lightwood.data.statistical_analysis(df, type_information, problem_definition)

    return DataAnalysis(
        type_information=type_information,
        statistical_analysis=statistical_analysis
    )