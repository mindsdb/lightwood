import pandas as pd
from lightwood.api.types import DataAnalysis, ProblemDefinition
import importlib
from lightwood.api.generate_predictor import generate_predictor
import lightwood
from lightwood.api.predictor import PredictorInterface
import os

def make_predictor(df: pd.DataFrame, problem_definition_dict: dict) -> PredictorInterface:
    predictor_class_str = generate_predictor(ProblemDefinition.from_dict(problem_definition_dict), df)

    try:
        os.remove('dynamic_predictor.py')
    except Exception:
        pass

    with open('dynamic_predictor.py', 'w') as fp:
        fp.write(predictor_class_str)

    predictor_class = importlib.import_module('dynamic_predictor').Predictor
    predictor = predictor_class()

    return predictor


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