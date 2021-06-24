from typing import Dict
import numpy as np
from lightwood.api.types import LightwoodConfig, TypeInformation, StatisticalAnalysis, Feature, Output, ProblemDefinition
from lightwood.api import dtype


trainable_encoders = ('TsRnnEncoder', 'PretrainedLangEncoder', 'CategoricalAutoEncoder')


def lookup_encoder(col_dtype: dtype, is_target: bool):
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
        dtype.quantity: 'NumericEncoder',
    }

    target_encoder_lookup_override = {
        dtype.rich_text: 'VocabularyEncoder',
        dtype.categorical: 'OneHotEncoder'
    }

    encoder_dict = {
        'object': encoder_lookup[col_dtype],
        'config_args': {},
        'dynamic_args': {}
    }

    if is_target:
        encoder_dict['dynamic_args'] = {'is_target': 'True'}
        if col_dtype in target_encoder_lookup_override:
            encoder_dict['object'] = target_encoder_lookup_override[col_dtype]
        if col_dtype in (dtype.categorical, dtype.binary):
            encoder_dict['config_args'] = {'target_class_distribution': 'statistical_analysis.target_class_distribution'}

    # Set arguments for the encoder
    if encoder_dict['object'] == 'PretrainedLangEncoder' and not is_target:
        encoder_dict['config_args']['output_type'] = 'output.data_dtype'

    if encoder_dict['object'] in trainable_encoders:
        encoder_dict['config_args']['stop_after'] = 'problem_definition.seconds_per_encoder'

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
            {
                
                'object': 'Neural',
                'config_args': {
                    'stop_after': 'problem_definition.seconds_per_model',
                    'timeseries_settings': 'problem_definition.timeseries_settings'
                },
                'dynamic_args': {
                    'target': 'self.target',
                    'dtype_dict': 'self.dtype_dict',
                    'input_cols': 'self.input_cols',
                    'target_encoder': 'self.encoders[self.target]'
                }
            },
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
                'models': 'self.models'
            }
        }
    )

    output.encoder = lookup_encoder(type_information.dtypes[target], True)

    features: Dict[str, Feature] = {}
    for col_name, col_dtype in type_information.dtypes.items():
        if col_name not in type_information.identifiers and col_dtype not in (dtype.invalid, dtype.empty) and col_name != target:
            feature = Feature(
                name=col_name,
                data_dtype=col_dtype,
                encoder=lookup_encoder(col_dtype, False),
                dependency=[]
            )
            features[col_name] = feature

    # @TODO: Only import the minimal amount of things we need
    imports = [
        'from lightwood.model import LightGBM',
        'from lightwood.model import Neural',
        'from lightwood.ensemble import BestOf',
        'from lightwood.data import cleaner',
        'from lightwood.data import transform_timeseries',
        'from lightwood.data import splitter',
        'from lightwood.analysis import model_analyzer, explain',
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
    timeseries_transformer = None
    if problem_definition.timeseries_settings.is_timeseries:
        timeseries_transformer = {
            'object': 'transform_timeseries',
            'config_args': {
                'timeseries_settings': 'problem_definition.timeseries_settings'
            },
            'dynamic_args': {
                'data': 'data'
            }
        }
    
    # Decide on the accuracy functions to use
    if output.data_dtype in [dtype.integer, dtype.float]:
        accuracy_functions = ['r2_score']
    elif output.data_dtype == dtype.categorical:
        accuracy_functions = ['balanced_accuracy_score']
    elif output.data_dtype == dtype.tags:
        accuracy_functions = ['balanced_accuracy_score']
    elif output.data_dtype == dtype.array:
        accuracy_functions = ['evaluate_array_accuracy']
    else:
        accuracy_functions = ['accuracy_score']
    
    if problem_definition.time_aim is None and (problem_definition.seconds_per_model is None or problem_definition.seconds_per_encoder is None):
        problem_definition.time_aim = 800 + statistical_analysis.nr_rows

    if problem_definition.time_aim is not None:
        # Should only be featurs wi2+np.log(nr_features)/5th trainable encoders
        nr_features = len([x for x in features.values() if x.encoder['object'] in trainable_encoders])
        nr_models = len(output.models)
        encoder_time_budget_pct = max(3.3 / 5, 1.5 + np.log(nr_features) / 5)
        problem_definition.seconds_per_encoder = problem_definition.time_aim * (encoder_time_budget_pct / nr_features)
        problem_definition.seconds_per_model = problem_definition.time_aim * ((1 / encoder_time_budget_pct) / nr_models)

    return LightwoodConfig(
        cleaner={
            'object': 'cleaner',
            'config_args': {
                'pct_invalid': 'problem_definition.pct_invalid',
                'ignore_features': 'problem_definition.ignore_features',
                'identifiers': 'identifiers'
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
                'stats_info': 'statistical_analysis',
                'ts_cfg': 'problem_definition.timeseries_settings',
                'accuracy_functions': 'accuracy_functions'
            },
            'dynamic_args': {
                'predictor': 'self.ensemble',
                'data': 'test_data',
                'target': 'self.target',
                'disable_column_importance': 'True',
                'dtype_dict': 'self.dtype_dict',
                'fixed_significance': None,
                'positive_domain': False,
            }
        },

        explainer={
            'object': 'explain',
            'config_args': {
                'timeseries_settings': 'problem_definition.timeseries_settings',
                'positive_domain': 'problem_definition.positive_domain',
                'fixed_confidence': 'problem_definition.fixed_confidence',
                'anomaly_detection': 'problem_definition.anomaly_detection',
                'anomaly_error_rate': 'problem_definition.anomaly_error_rate',
                'anomaly_cooldown': 'problem_definition.anomaly_cooldown'
            },
            'dynamic_args': {
                'data': 'data',
                'predictions': 'df',
                'analysis': 'self.runtime_analyzer',
                'target_name': 'self.target',
                'target_dtype': 'self.dtype_dict[self.target]',
            }
        },
        features=features,
        output=output,
        imports=imports,
        problem_definition=problem_definition,
        statistical_analysis=statistical_analysis,
        identifiers=type_information.identifiers,
        timeseries_transformer=timeseries_transformer,
        accuracy_functions=accuracy_functions
    )
