from lightwood.helpers.templating import call, inline_dict, align
from lightwood.api.types import ProblemDefinition
import lightwood
from lightwood.api import JsonML
import pandas as pd
import autopep8


def add_implicit_values(json_ml: JsonML) -> str:
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

    for feature in [json_ml.output, *json_ml.features.values()]:
        encoder_import = feature.encoder['object']
        imports.append(f'from lightwood.encoder import {encoder_import}')

    json_ml.imports.extend(imports)

    return json_ml


def generate_predictor_code(json_ml: JsonML) -> str:
    json_ml = add_implicit_values(json_ml)

    predictor_code = ''

    imports = '\n'.join(json_ml.imports)

    encoder_dict = {json_ml.output.name: call(json_ml.output.encoder, json_ml)}
    dependency_dict = {}
    dtype_dict = {json_ml.output.name: f"""'{json_ml.output.data_dtype}'"""}

    for col_name, feature in json_ml.features.items():
        encoder_dict[col_name] = call(feature.encoder, json_ml)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    input_cols = ','.join([f"""'{feature.name}'""" for feature in json_ml.features.values()])

    ts_code = ''
    if json_ml.timeseries_transformer is not None:
        ts_code = f"""
log.info('Transforming timeseries data')
data = {call(json_ml.timeseries_transformer, json_ml)}
"""

    learn_body = f"""
# How columns are encoded
self.encoders = {inline_dict(encoder_dict)}
# Which column depends on which
self.dependencies = {inline_dict(dependency_dict)}
# The type of each column
self.dtype_dict = {inline_dict(dtype_dict)}
#
self.input_cols = [{input_cols}]

log.info('Cleaning the data')
data = {call(json_ml.cleaner, json_ml)}

{ts_code}

nfolds = {json_ml.problem_definition.nfolds}
log.info(f'Splitting the data into {{nfolds}} folds')
folds = {call(json_ml.splitter, json_ml)}

log.info('Preparing the encoders')

parallel_preped_encoders = mut_method_call({{col_name: [encoder, pd.concat(folds[0:nfolds-1])[col_name], 'prepare'] for col_name, encoder in self.encoders.items() if not encoder.is_nn_encoder}})

seq_preped_encoders = {{}}
for col_name, encoder in self.encoders.items():
    if encoder.is_nn_encoder:
        encoder.prepare(pd.concat(folds[0:nfolds-1])[col_name])

for col_name, encoder in parallel_preped_encoders.items():
    self.encoders[col_name] = encoder

log.info('Featurizing the data')
encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)
train_data = encoded_ds_arr[0:nfolds-1]
test_data = encoded_ds_arr[nfolds-1]

log.info('Training the models')
self.models = [{', '.join([call(x, json_ml) for x in json_ml.output.models])}]
for model in self.models:
    model.fit(train_data)

log.info('Ensembling the model')
self.ensemble = {call(json_ml.output.ensemble, json_ml)}

log.info('Analyzing the ensemble')
self.model_analysis, self.runtime_analyzer = {call(json_ml.analyzer, json_ml)}
"""
    learn_body = align(learn_body, 2)

    predict_body = f"""
log.info('Cleaning the data')
data = {call(json_ml.cleaner, json_ml)}

{ts_code}

encoded_ds = lightwood.encode(self.encoders, data, self.target)
df = self.ensemble(encoded_ds)
insights = {call(json_ml.explainer, json_ml)}
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

    def __init__(self):
        seed()
        self.target = '{json_ml.output.name}'

    def learn(self, data: pd.DataFrame) -> None:
{learn_body}

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
{predict_body}
"""

    return predictor_code


def generate_predictor(problem_definition: ProblemDefinition = None, data: pd.DataFrame = None, json_ml: JsonML = None) -> str:
    if json_ml is None:
        type_information = lightwood.data.infer_types(data, problem_definition.pct_invalid)
        statistical_analysis = lightwood.data.statistical_analysis(data, type_information, problem_definition)
        json_ml = lightwood.generate_json_ml(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)
        
        print(json_ml.to_json())
        exit()

    predictor_code = generate_predictor_code(json_ml)

    predictor_code = autopep8.fix_code(predictor_code)  # Note: ~3s overhead, might be more depending on source complexity, should try a few more examples and make a decision

    return predictor_code
