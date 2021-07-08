from lightwood.helpers.templating import call, inline_dict, align
from lightwood.api.generate_json_ai import lookup_encoder
from lightwood.api import JsonAI
import autopep8


def add_implicit_values(json_ai: JsonAI) -> JsonAI:
    imports = [
        'from lightwood.model import LightGBM',
        'from lightwood.model import Neural',
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
        imports.append(f'from lightwood.encoder import TimeSeriesPlainEncoder')

    json_ai.imports.extend(imports)

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    json_ai = add_implicit_values(json_ai)

    predictor_code = ''

    imports = '\n'.join(json_ai.imports)

    encoder_dict = {json_ai.output.name: call(json_ai.output.encoder, json_ai)}
    dependency_dict = {}
    dtype_dict = {json_ai.output.name: f"""'{json_ai.output.data_dtype}'"""}

    for col_name, feature in json_ai.features.items():
        encoder_dict[col_name] = call(feature.encoder, json_ai)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    if json_ai.problem_definition.timeseries_settings.use_previous_target:
        col_name = f'__mdb_ts_previous_{json_ai.output.name}'
        json_ai.problem_definition.timeseries_settings.target_type = json_ai.output.data_dtype
        encoder_dict[col_name] = call(lookup_encoder(json_ai.output.data_dtype,
                                                     col_name,
                                                     json_ai.problem_definition.timeseries_settings,
                                                     json_ai.statistical_analysis,
                                                     is_target=False),
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
        ts_encoder_code = f"""
if type(encoder) in __ts_encoders__:
    kwargs['ts_analysis'] = self.ts_analysis
"""

    if json_ai.problem_definition.timeseries_settings.is_timeseries:
        ts_target_code = f"""
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
        log.info(f'Encoder preping dict lenght of: {{len(encoder_preping_dict)}}')

parallel_preped_encoders = mut_method_call(encoder_preping_dict)

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
        encoder.prepare(priming_data[col_name], **kwargs)

for col_name, encoder in parallel_preped_encoders.items():
    self.encoders[col_name] = encoder
    {align(ts_target_code, 1)}
"""
    dataprep_body = align(dataprep_body, 2)

    learn_body = f"""
log.info('Featurizing the data')
encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)
train_data = encoded_ds_arr[0:nfolds-1]
test_data = encoded_ds_arr[nfolds-1]

log.info('Training the models')
self.models = [{', '.join([call(x, json_ai) for x in json_ai.output.models])}]
for model in self.models:
    model.fit(train_data)

log.info('Ensembling the model')
self.ensemble = {call(json_ai.output.ensemble, json_ai)}

log.info('Analyzing the ensemble')
self.model_analysis, self.runtime_analyzer = {call(json_ai.analyzer, json_ai)}

# Partially fit the model on the reamining of the data, data is precious, we mustn't loss one bit
for model in self.models:
    model.partial_fit([test_data])
"""
    learn_body = align(learn_body, 2)

    predict_body = f"""
self.mode = 'predict'
log.info('Cleaning the data')
data = {call(json_ai.cleaner, json_ai)}

{ts_transform_code}

encoded_ds = lightwood.encode(self.encoders, data, self.target)
df = self.ensemble(encoded_ds)
insights = {call(json_ai.explainer, json_ai)}
return insights
"""
    predict_body = align(predict_body, 2)

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
