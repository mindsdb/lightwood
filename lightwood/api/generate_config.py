from typing import Dict
import numpy as np
from lightwood.api.types import LightwoodConfig, TypeInformation, StatisticalAnalysis, Feature, Output, ProblemDefinition
from lightwood.api import dtype


def lookup_encoder(col_dtype: dtype, is_target: bool, output: Output):
    encoder_lookup = {
        dtype.integer: 'NumericEncoder',
        dtype.float: 'NumericEncoder',
        dtype.binary: 'OneHotEncoder',
        dtype.categorical: 'CategoricalAutoEncoder',
        dtype.tags: 'MultiHotEncoder',
        dtype.date: 'DatetimeEncoder',
        dtype.datetime: 'DatetimeEncoder',
        dtype.image: 'Img2VecEncoder',
        dtype.rich_text: 'PretrainedLangEncoder',
        dtype.short_text: 'ShortTextEncoder',
        dtype.array: 'TsRnnEncoder',
    }

    target_encoder_lookup_override = {
        dtype.rich_text: 'VocabularyEncoder'
    }

    encoder_dict = {
        'object': encoder_lookup[col_dtype],
        'config_args': {},
        'dynamic_args': {}
    }
    if is_target:
        if col_dtype in target_encoder_lookup_override:
            encoder_dict['object'] = target_encoder_lookup_override[col_dtype]

    # Set arguments for the encoder
    if encoder_dict['object'] == 'PretrainedLangEncoder' and not is_target:
        encoder_dict['config_args']['output_type'] = 'output.data_dtype'

    return encoder_dict


def populate_problem_definition(type_information: TypeInformation, statistical_analysis: StatisticalAnalysis, problem_definition: ProblemDefinition) -> ProblemDefinition:
    if problem_definition.seconds_per_model is None:
        problem_definition.seconds_per_model = max(100, statistical_analysis.nr_rows / 20) * np.sum([4 if x in [dtype.rich_text, dtype.short_text, dtype.array, dtype.video, dtype.audio, dtype.image] else 1 for x in type_information.dtypes.values()])
    
    return problem_definition


def generate_config(type_information: TypeInformation, statistical_analysis: StatisticalAnalysis, problem_definition: ProblemDefinition) -> LightwoodConfig:

    problem_definition = populate_problem_definition(type_information, statistical_analysis, problem_definition)
    target = problem_definition.target

    output = Output(
        name=target,
        data_dtype=type_information.dtypes[target],
        encoder=None,
        models=[
            '''
            {
                
                'object': 'Neural',
                'config_args': {
                    'stop_after': 'problem_definition.seconds_per_model',
                    'timeseries_settings': 'problem_definition.timeseries_settings'
                },
                'dynamic_args': {
                    'target': 'self.target',
                    'dtype_dict': 'self.dtype_dict',
                    'input_cols': 'self.input_cols'
                }
            },
            '''
            {
                'object': 'LightGBM',
                'config_args': {
                    'stop_after': 'problem_definition.seconds_per_model'
                },
                'dynamic_args': {
                    'target': 'self.target',
                    'dtype_dict': 'self.dtype_dict',
                    'input_cols': 'self.input_cols'
                }
            }
        ],
        ensemble={
            'object': 'BestOf',
            'config_args': {},
            'dynamic_args': {
                'data': 'test_data',
                'modles': 'self.models'
            }
        }
    )

    output.encoder = lookup_encoder(type_information.dtypes[target], True, output)

    features: Dict[str, Feature] = {}
    for col_name, col_dtype in type_information.dtypes.items():
        if type_information.identifiers[col_name] is None and col_dtype not in (dtype.invalid, dtype.empty) and col_name != target:
            feature = Feature(
                name=col_name,
                data_dtype=col_dtype,
                encoder=lookup_encoder(col_dtype, False, output),
                dependency=[]
            )
            features[col_name] = feature

    # @TODO: Only import the minimal amount of things we need
    imports = [
        'from lightwood.model import LightGBM',
        'from lightwood.model import Neural',
        'from lightwood.ensemble import BestOf',
        'from lightwood.data import cleaner',
        'from lightwood.data import splitter',
        'from lightwood.analysis import model_analyzer',
        'from sklearn.metrics import r2_score, balanced_accuracy_score, accuracy_score',
        'import pandas as pd',
        'from mindsdb_datasources import DataSource',
        'from lightwood.helpers.seed import seed',
        'from lightwood.helpers.log import log',
        'import lightwood',
        'from lightwood.api import *',
        'from lightwood.model import BaseModel',
        'from lightwood.encoder import BaseEncoder',
        'from lightwood.ensemble import BaseEnsemble',
        'from typing import Dict, List',
        'from lightwood.helpers.parallelism import mut_method_call'
    ]

    for feature in [output, *features.values()]:
        encoder_import = feature.encoder['object']
        imports.append(f'from lightwood.encoder import {encoder_import}')

    imports = list(set(imports))
    return LightwoodConfig(
        cleaner={
            'object': 'cleaner',
            'config_args': {
                'pct_invalid': 'problem_definition.pct_invalid'
            },
            'dynamic_args': {
                'data': 'data',
                'dtype_dict': 'self.dtype_dict'
            }
        },
        splitter={
            'object': 'splitter',
            'config_args': {},
            'dynamic_args': {
                'data': 'data',
                'k': 'nfolds'
            }
        },
        analyzer={
            'object': 'model_analyzer',
            'config_args': {
                'encoded_data': 'test_data',
                'stats_info': 'statistical_analysis',
                'target': 'output',
                'features': 'features'
            },
            'dynamic_args': {
                'predictor': 'self.ensemble',
                'disable_column_importance': 'True'
            }
        },
        features=features,
        output=output,
        imports=imports,
        problem_definition=problem_definition,
        statistical_analysis=statistical_analysis
    )
