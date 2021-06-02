from typing import Dict
from lightwood.helpers.templating import call, inline_dict, align
from lightwood.api.types import ProblemDefinition
import lightwood
import pprint
from lightwood.api import LightwoodConfig
from mindsdb_datasources import DataSource


def generate_predictor_code(lightwood_config: LightwoodConfig) -> str:
    predictor_code = ''

    imports = '\n'.join(lightwood_config.imports)

    encoder_dict = {lightwood_config.output.name: call(lightwood_config.output.encoder, lightwood_config)}
    dependency_dict = {}
    dtype_dict = {lightwood_config.output.name: f"""'{lightwood_config.output.data_dtype}'"""}

    for col_name, feature in lightwood_config.features.items():
        encoder_dict[col_name] = call(feature.encoder,  lightwood_config)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    input_cols = ','.join([f"""'{feature.name}'""" for feature in lightwood_config.features.values()])

    learn_body = f"""
# How columns are encoded
self.encoders = {inline_dict(encoder_dict)}
# Which column depends on which
self.dependencies = {inline_dict(dependency_dict)}
# The type of each column
self.dtype_dict = {inline_dict(dtype_dict)}
#
self.input_cols = {[input_cols]}

log.info('Cleaning the data')
data = {call(lightwood_config.cleaner, lightwood_config)}

nfolds = {lightwood_config.problem_definition.nfolds}
log.info(f'Splitting the data into {{nfolds}} folds')
folds = {call(lightwood_config.splitter, lightwood_config)}

log.info('Preparing the encoders')
self.encoders = mut_method_call({{col_name: [encoder, pd.concat(folds[0:nfolds-1])[col_name], 'prepare'] for col_name, encoder in self.encoders.items()}})

log.info('Featurizing the data')
encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)

log.info('Training the models')
self.models = [{', '.join([call(x, lightwood_config) for x in lightwood_config.output.models])}]
for model in self.models:
    model.fit(encoded_ds_arr[0:nfolds-1])

log.info('Ensembling the model')
self.ensemble = {lightwood_config.output.ensemble}(self.models, encoded_ds_arr[nfolds-1])

log.info('Analyzing the ensemble')
# Add back when analysis works
self.confidence_model, self.predictor_analysis = {lightwood_config.analyzer}(self.ensemble, encoded_ds_arr[nfolds-1], folds[nfolds-1])
"""
    learn_body = align(learn_body, 2)

    predict_body = f"""
encoded_ds = lightwood.encode(self.encoders, data.df, self.target)
df = self.ensemble(encoded_ds)
return df
"""
    predict_body = align(predict_body, 2)

    predictor_code = f"""
{imports}

class Predictor():
    target: str
    models: List[BaseModel]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble

    def __init__(self):
        seed()
        self.target = '{lightwood_config.output.name}'

    def learn(self, data: DataSource) -> None:
{learn_body}

    def predict(self, data: DataSource) -> pd.DataFrame:
{predict_body}
"""

    return predictor_code


def generate_predictor(problem_definition: ProblemDefinition = None, datasource: DataSource = None, lightwood_config: LightwoodConfig = None) -> str:
    if lightwood_config is None:
        type_information = lightwood.data.infer_types(datasource, problem_definition.pct_invalid)
        statistical_analysis = lightwood.data.statistical_analysis(datasource, type_information, problem_definition)
        lightwood_config = lightwood.generate_config(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)

    predictor_code = generate_predictor_code(lightwood_config)
    print(lightwood_config.to_json())
    exit()
    return predictor_code
