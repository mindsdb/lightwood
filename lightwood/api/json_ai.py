from typing import Dict
from lightwood.helpers.templating import call, inline_dict, align
import autopep8
from lightwood.api import dtype
import numpy as np
from lightwood.api.types import JsonAI, TypeInformation, StatisticalAnalysis, Feature, Output, ProblemDefinition, TimeseriesSettings


trainable_encoders = ('PretrainedLangEncoder', 'CategoricalAutoEncoder', 'TimeSeriesEncoder', 'TimeSeriesPlainEncoder')
ts_encoders = ('TimeSeriesEncoder', 'TimeSeriesPlainEncoder', 'TsNumericEncoder')


def lookup_encoder(col_dtype: str, col_name: str, statistical_analysis: StatisticalAnalysis, is_target: bool, problem_defintion: ProblemDefinition, is_target_predicting_encoder: bool):
    tss = problem_defintion.timeseries_settings
    encoder_lookup = {
        dtype.integer: 'NumericEncoder',
        dtype.float: 'NumericEncoder',
        dtype.binary: 'BinaryEncoder',
        dtype.categorical: 'CategoricalAutoEncoder',
        dtype.tags: 'MultiHotEncoder',
        dtype.date: 'DatetimeEncoder',
        dtype.datetime: 'DatetimeEncoder',
        dtype.image: 'Img2VecEncoder',
        dtype.rich_text: 'PretrainedLangEncoder',
        dtype.short_text: 'ShortTextEncoder',
        dtype.array: 'TimeSeriesEncoder',
        dtype.quantity: 'NumericEncoder',
    }

    target_encoder_lookup_override = {
        dtype.rich_text: 'VocabularyEncoder',
        dtype.categorical: 'OneHotEncoder'
    }

    encoder_dict = {
        'object': encoder_lookup[col_dtype],
        'dynamic_args': {}
    }

    if col_dtype == dtype.categorical and len(statistical_analysis.histograms) < 100:
        encoder_dict = {
            'object': 'OneHotEncoder',
            'dynamic_args': {}
        }

    if is_target:
        encoder_dict['dynamic_args'] = {'is_target': 'True'}
        if col_dtype in target_encoder_lookup_override:
            encoder_dict['object'] = target_encoder_lookup_override[col_dtype]
        if col_dtype in (dtype.categorical, dtype.binary):
            if problem_defintion.unbias_target:
                encoder_dict['dynamic_args'] = {'target_class_distribution': '$statistical_analysis.target_class_distribution'}

    if tss.is_timeseries:
        gby = tss.group_by if tss.group_by is not None else []
        if col_name in tss.order_by + tss.historical_columns:
            encoder_dict['object'] = 'TimeSeriesEncoder'
            encoder_dict['dynamic_args']['original_type'] = f'"{col_dtype}"'
            encoder_dict['dynamic_args']['target'] = "self.target"
            encoder_dict['dynamic_args']['grouped_by'] = f"{gby}"
        if is_target:
            if col_dtype in [dtype.integer, dtype.float]:
                encoder_dict['dynamic_args']['grouped_by'] = f"{gby}"
                encoder_dict['object'] = 'TsNumericEncoder'
            if tss.nr_predictions > 1:
                encoder_dict['dynamic_args']['grouped_by'] = f"{gby}"
                encoder_dict['dynamic_args']['timesteps'] = f"{tss.nr_predictions}"
                encoder_dict['object'] = 'TsArrayNumericEncoder'
        if '__mdb_ts_previous' in col_name:
            encoder_dict['object'] = 'TimeSeriesPlainEncoder'
            encoder_dict['dynamic_args']['original_type'] = f'"{tss.target_type}"'
            encoder_dict['dynamic_args']['window'] = f'{tss.window}'

    # Set arguments for the encoder
    if encoder_dict['object'] == 'PretrainedLangEncoder' and not is_target:
        encoder_dict['dynamic_args']['output_type'] = '$output.data_dtype'

    if encoder_dict['object'] in trainable_encoders:
        encoder_dict['dynamic_args']['stop_after'] = '$problem_definition.seconds_per_encoder'

    if is_target_predicting_encoder:
        encoder_dict['dynamic_args']['embed_mode'] = 'False'

    return encoder_dict


def generate_json_ai(type_information: TypeInformation, statistical_analysis: StatisticalAnalysis, problem_definition: ProblemDefinition) -> JsonAI:
    target = problem_definition.target
    input_cols = []
    for col_name, col_dtype in type_information.dtypes.items():
        if col_name not in type_information.identifiers and col_dtype not in (dtype.invalid, dtype.empty) and col_name != target:
            input_cols.append(col_name)

    is_target_predicting_encoder = False
    # Single text column classification
    if len(input_cols) == 1 and type_information.dtypes[input_cols[0]] in (dtype.rich_text) and type_information.dtypes[target] in (dtype.categorical, dtype.binary):
        is_target_predicting_encoder = True

    if is_target_predicting_encoder:
        models = [{
            'object': 'Unit',
            'dynamic_args': {
                'target_encoder': 'self.encoders[self.target]',
                'stop_after': '$problem_definition.seconds_per_model'
            }
        }]
    else:
        models = [{
                'object': 'Neural',
                'dynamic_args': {
                    'target_encoder': 'self.encoders[self.target]',
                    'target': 'self.target',
                    'dtype_dict': 'self.dtype_dict',
                    'input_cols': 'self.input_cols',
                    'net': f'"DefaultNet"' if not problem_definition.timeseries_settings.is_timeseries else f'"ArNet"',
                    'fit_on_dev': True,
                    'stop_after': '$problem_definition.seconds_per_model',
                    'timeseries_settings': '$problem_definition.timeseries_settings',
                }

        }]

    if not problem_definition.timeseries_settings.is_timeseries or \
            problem_definition.timeseries_settings.nr_predictions <= 1:
        models.extend([{
                'object': 'LightGBM',
                'dynamic_args': {
                    'stop_after': '$problem_definition.seconds_per_model',
                    'target': 'self.target',
                    'dtype_dict': 'self.dtype_dict',
                    'input_cols': 'self.input_cols',
                    'fit_on_dev': True
                }
            },
            {
                'object': 'Regression',
                'dynamic_args': {
                    'stop_after': '$problem_definition.seconds_per_model',
                    'target_encoder': 'self.encoders[self.target]'
                }
            }
        ])
    elif problem_definition.timeseries_settings.nr_predictions > 1:
        models.extend([{
            'object': 'LightGBMArray',
            'dynamic_args': {
                'target': 'self.target',
                'dtype_dict': 'self.dtype_dict',
                'input_cols': 'self.input_cols',
                'fit_on_dev': True,
                'stop_after': '$problem_definition.seconds_per_model',
                'n_ts_predictions': '$problem_definition.timeseries_settings.nr_predictions'
            }
        },
            {
            'object': 'SkTime',
            'dynamic_args': {
                'stop_after': '$problem_definition.seconds_per_model',
                'n_ts_predictions': '$problem_definition.timeseries_settings.nr_predictions',
                'target': 'self.target',
                'dtype_dict': 'self.dtype_dict',
                'ts_analysis': 'self.ts_analysis'
            },
            }
        ])

    models = [models[-1]]

    output = Output(
        name=target,
        data_dtype=type_information.dtypes[target],
        encoder=None,
        models=models,
        ensemble={
            'object': 'BestOf',
            'dynamic_args': {
                'accuracy_functions': '$accuracy_functions',
                'target': 'self.target',
                'data': 'test_data',
                'models': 'self.models'
            }
        }
    )

    output.encoder = lookup_encoder(type_information.dtypes[target], target, statistical_analysis, True, problem_definition, False)

    features: Dict[str, Feature] = {}
    for col_name in input_cols:
        col_dtype = type_information.dtypes[col_name]
        dependency = []
        encoder = lookup_encoder(col_dtype, col_name, statistical_analysis, False, problem_definition, is_target_predicting_encoder)

        if problem_definition.timeseries_settings.is_timeseries and encoder['object'] in ts_encoders:
            if problem_definition.timeseries_settings.group_by is not None:
                for group in problem_definition.timeseries_settings.group_by:
                    dependency.append(group)

            if problem_definition.timeseries_settings.use_previous_target:
                dependency.append(f'__mdb_ts_previous_{target}')

        feature = Feature(
            name=col_name,
            data_dtype=col_dtype,
            encoder=encoder,
            dependency=dependency
        )
        features[col_name] = feature

    timeseries_transformer = None
    if problem_definition.timeseries_settings.is_timeseries:
        timeseries_transformer = {
            'object': 'transform_timeseries',
            'dynamic_args': {
                'timeseries_settings': '$problem_definition.timeseries_settings',
                'data': 'data',
                'dtype_dict': 'self.dtype_dict',
                'target': 'self.target',
                'mode': 'self.mode'
            }
        }

        timeseries_analyzer = {
            'object': 'timeseries_analyzer',
            'dynamic_args': {
                'timeseries_settings': '$problem_definition.timeseries_settings',
                'data': 'data',
                'dtype_dict': 'self.dtype_dict',
                'target': 'self.target'
            }
        }

        if problem_definition.timeseries_settings.nr_predictions > 1:
            output.data_dtype = dtype.array
    else:
        timeseries_analyzer = None

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
        problem_definition.time_aim = 1000 + np.log(statistical_analysis.nr_rows / 10 + 1) * np.sum([4 if x in [dtype.rich_text, dtype.short_text, dtype.array, dtype.video, dtype.audio, dtype.image] else 1 for x in type_information.dtypes.values()]) * 200

    if problem_definition.time_aim is not None:
        nr_trainable_encoders = len([x for x in features.values() if x.encoder['object'] in trainable_encoders])
        nr_models = len(output.models)
        encoder_time_budget_pct = max(3.3 / 5, 1.5 + np.log(nr_trainable_encoders + 1) / 5)

        if nr_trainable_encoders == 0:
            problem_definition.seconds_per_encoder = 0
        else:
            problem_definition.seconds_per_encoder = int(problem_definition.time_aim * (encoder_time_budget_pct / nr_trainable_encoders))
        problem_definition.seconds_per_model = int(problem_definition.time_aim * ((1 / encoder_time_budget_pct) / nr_models))

    return JsonAI(
        cleaner={
            'object': 'cleaner',
            'dynamic_args': {
                'pct_invalid': '$problem_definition.pct_invalid',
                'ignore_features': '$problem_definition.ignore_features',
                'identifiers': '$identifiers',
                'data': 'data',
                'dtype_dict': 'self.dtype_dict',
                'target': 'self.target',
                'mode': 'self.mode'
            }
        },
        splitter={
            'object': 'splitter',
            'dynamic_args': {
                'tss': '$problem_definition.timeseries_settings',
                'data': 'data',
                'k': 'nfolds'
            }
        },
        analyzer={
            'object': 'model_analyzer',
            'dynamic_args': {
                'stats_info': '$statistical_analysis',
                'ts_cfg': '$problem_definition.timeseries_settings',
                'accuracy_functions': '$accuracy_functions',
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
            'dynamic_args': {
                'timeseries_settings': '$problem_definition.timeseries_settings',
                'positive_domain': '$problem_definition.positive_domain',
                'fixed_confidence': '$problem_definition.fixed_confidence',
                'anomaly_detection': '$problem_definition.anomaly_detection',
                'anomaly_error_rate': '$problem_definition.anomaly_error_rate',
                'anomaly_cooldown': '$problem_definition.anomaly_cooldown',
                'data': 'data',
                'encoded_data': 'encoded_data',
                'predictions': 'df',
                'analysis': 'self.runtime_analyzer',
                'ts_analysis': 'self.ts_analysis' if problem_definition.timeseries_settings.is_timeseries else None,
                'target_name': 'self.target',
                'target_dtype': 'self.dtype_dict[self.target]',
            }
        },
        features=features,
        output=output,
        imports=[],
        problem_definition=problem_definition,
        statistical_analysis=statistical_analysis,
        identifiers=type_information.identifiers,
        timeseries_transformer=timeseries_transformer,
        timeseries_analyzer=timeseries_analyzer,
        accuracy_functions=accuracy_functions
    )


def add_implicit_values(json_ai: JsonAI) -> JsonAI:
    imports = [
        'from lightwood.model import Neural',
        'from lightwood.model import LightGBM',
        'from lightwood.model import LightGBMArray',
        'from lightwood.model import SkTime',
        'from lightwood.model import Unit',
        'from lightwood.model import Regression',
        'from lightwood.ensemble import BestOf',
        'from lightwood.data import cleaner',
        'from lightwood.data import transform_timeseries, timeseries_analyzer',
        'from lightwood.data import splitter',
        'from lightwood.analysis import model_analyzer, explain',
        'from sklearn.metrics import r2_score, balanced_accuracy_score, accuracy_score',
        'import pandas as pd',
        'from lightwood.helpers.seed import seed',
        'from lightwood.helpers.log import log',
        'import lightwood',
        'from lightwood.api import *',
        'from lightwood.model import BaseModel',
        'from lightwood.encoder import BaseEncoder, __ts_encoders__',
        'from lightwood.ensemble import BaseEnsemble',
        'from typing import Dict, List',
        'from lightwood.helpers.parallelism import mut_method_call'
    ]

    for feature in [json_ai.output, *json_ai.features.values()]:
        encoder_import = feature.encoder['object']
        imports.append(f'from lightwood.encoder import {encoder_import}')

    if json_ai.problem_definition.timeseries_settings.use_previous_target:
        imports.append('from lightwood.encoder import TimeSeriesPlainEncoder')

    json_ai.imports.extend(imports)

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    json_ai = add_implicit_values(json_ai)

    encoder_dict = {json_ai.output.name: call(json_ai.output.encoder, json_ai)}
    dependency_dict = {}
    dtype_dict = {json_ai.output.name: f"""'{json_ai.output.data_dtype}'"""}

    for col_name, feature in json_ai.features.items():
        encoder_dict[col_name] = call(feature.encoder, json_ai)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    # @TODO: Move into json-ai creation function (I think? Maybe? Let's discuss)
    if json_ai.problem_definition.timeseries_settings.use_previous_target:
        col_name = f'__mdb_ts_previous_{json_ai.output.name}'
        json_ai.problem_definition.timeseries_settings.target_type = json_ai.output.data_dtype
        encoder_dict[col_name] = call(lookup_encoder(json_ai.output.data_dtype,
                                                     col_name,
                                                     json_ai.statistical_analysis,
                                                     False,
                                                     json_ai.problem_definition,
                                                     False,
                                                     ),
                                      json_ai)
        dependency_dict[col_name] = []
        dtype_dict[col_name] = f"""'{json_ai.output.data_dtype}'"""

    input_cols = ','.join([f"""'{feature.name}'""" for feature in json_ai.features.values()])

    ts_transform_code = ''
    ts_analyze_code = ''
    ts_encoder_code = ''
    if json_ai.timeseries_transformer is not None:
        ts_transform_code = f"""
log.info('Transforming timeseries data')
data = {call(json_ai.timeseries_transformer, json_ai)}
"""
        ts_analyze_code = f"""
self.ts_analysis = {call(json_ai.timeseries_analyzer, json_ai)}
"""

    if json_ai.timeseries_analyzer is not None:
        ts_encoder_code = """
if type(encoder) in __ts_encoders__:
    kwargs['ts_analysis'] = self.ts_analysis
"""

    if json_ai.problem_definition.timeseries_settings.is_timeseries:
        ts_target_code = """
if encoder.is_target:
    encoder.normalizers = self.ts_analysis['target_normalizers']
    encoder.group_combinations = self.ts_analysis['group_combinations']
"""
    else:
        ts_target_code = ''

    dataprep_body = f"""
self.mode = 'train'
# How columns are encoded
self.encoders = {inline_dict(encoder_dict)}
# Which column depends on which
self.dependencies = {inline_dict(dependency_dict)}
# The type of each column
self.dtype_dict = {inline_dict(dtype_dict)}
#
self.input_cols = [{input_cols}]

log.info('Cleaning the data')
data = {call(json_ai.cleaner, json_ai)}

{ts_transform_code}
{ts_analyze_code}

nfolds = {json_ai.problem_definition.nfolds}
log.info(f'Splitting the data into {{nfolds}} folds')
folds = {call(json_ai.splitter, json_ai)}

log.info('Preparing the encoders')

encoder_preping_dict = {{}}
enc_preping_data = pd.concat(folds[0:nfolds-1])
for col_name, encoder in self.encoders.items():
    if not encoder.is_nn_encoder:
        encoder_preping_dict[col_name] = [encoder, enc_preping_data[col_name], 'prepare']
        log.info(f'Encoder preping dict length of: {{len(encoder_preping_dict)}}')

parallel_preped_encoders = mut_method_call(encoder_preping_dict)
for col_name, encoder in parallel_preped_encoders.items():
    self.encoders[col_name] = encoder

if self.target not in parallel_preped_encoders:
    self.encoders[self.target].prepare(enc_preping_data[self.target])

for col_name, encoder in self.encoders.items():
    if encoder.is_nn_encoder:
        priming_data = pd.concat(folds[0:nfolds-1])
        kwargs = {{}}
        if self.dependencies[col_name]:
            kwargs['dependency_data'] = {{}}
            for col in self.dependencies[col_name]:
                kwargs['dependency_data'][col] = {{
                    'original_type': self.dtype_dict[col],
                    'data': priming_data[col]
                }}
            {align(ts_encoder_code, 3)}
        
        # This assumes target  encoders are also prepared in parallel, might not be true
        if hasattr(encoder, 'uses_target'):
            kwargs['encoded_target_values'] = parallel_preped_encoders[self.target].encode(priming_data[self.target])

        encoder.prepare(priming_data[col_name], **kwargs)

    {align(ts_target_code, 1)}
"""
    dataprep_body = align(dataprep_body, 2)

    learn_body = f"""
log.info('Featurizing the data')
encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)
train_data = encoded_ds_arr[0:int(nfolds*0.9)]
test_data = encoded_ds_arr[int(nfolds*0.9):]

log.info('Training the models')
self.models = [{', '.join([call(x, json_ai) for x in json_ai.output.models])}]
trained_models = []
for model in self.models:
    try:
        model.fit(train_data)
        trained_models.append(model)
    except Exception as e:
        if {json_ai.problem_definition.strict_mode}:
            raise e
self.models = trained_models

log.info('Ensembling the model')
self.ensemble = {call(json_ai.output.ensemble, json_ai)}

log.info('Analyzing the ensemble')
self.model_analysis, self.runtime_analyzer = {call(json_ai.analyzer, json_ai)}

# Partially fit the model on the reamining of the data, data is precious, we mustn't loss one bit
for model in self.models:
    if {json_ai.problem_definition.fit_on_validation}:
        model.partial_fit(test_data, train_data)
"""
    learn_body = align(learn_body, 2)

    predict_body = f"""
self.mode = 'predict'
log.info('Cleaning the data')
data = {call(json_ai.cleaner, json_ai)}

{ts_transform_code}

encoded_ds = lightwood.encode(self.encoders, data, self.target)
encoded_data = encoded_ds.get_encoded_data(include_target=False)

df = self.ensemble(encoded_ds)
insights = {call(json_ai.explainer, json_ai)}
return insights
"""
    predict_body = align(predict_body, 2)

    imports = '\n'.join(json_ai.imports)
    predictor_code = f"""
{imports}
from lightwood.api import PredictorInterface


class Predictor(PredictorInterface):
    target: str
    models: List[BaseModel]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble
    mode: str

    def __init__(self):
        seed()
        self.target = '{json_ai.output.name}'
        self.mode = 'innactive'

    def learn(self, data: pd.DataFrame) -> None:
{dataprep_body}
{learn_body}

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
{predict_body}
"""

    if len(predictor_code) < 5000:
        predictor_code = autopep8.fix_code(predictor_code)

    return predictor_code


def validate_json_ai(json_ai: JsonAI) -> bool:
    from lightwood.api.high_level import predictor_from_code, code_from_json_ai
    try:
        predictor_from_code(code_from_json_ai(json_ai))
        return True
    except Exception:
        return False
