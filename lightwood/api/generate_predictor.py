from lightwood.helpers.templating import call, inline_dict, align
from lightwood.api.types import ProblemDefinition
import lightwood
from lightwood.api import LightwoodConfig
import pandas as pd


def generate_predictor_code(lightwood_config: LightwoodConfig) -> str:
    predictor_code = ''

    imports = '\n'.join(lightwood_config.imports)

    encoder_dict = {lightwood_config.output.name: call(lightwood_config.output.encoder, lightwood_config)}
    dependency_dict = {}
    dtype_dict = {lightwood_config.output.name: f"""'{lightwood_config.output.data_dtype}'"""}

    for col_name, feature in lightwood_config.features.items():
        encoder_dict[col_name] = call(feature.encoder, lightwood_config)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    input_cols = ','.join([f"""'{feature.name}'""" for feature in lightwood_config.features.values()])

    ts_code = ''
    if lightwood_config.timeseries_transformer is not None:
        ts_code = f"""
log.info('Transforming timeseries data')
data = {call(lightwood_config.timeseries_transformer, lightwood_config)}
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
data = {call(lightwood_config.cleaner, lightwood_config)}

{ts_code}

nfolds = {lightwood_config.problem_definition.nfolds}
log.info(f'Splitting the data into {{nfolds}} folds')
folds = {call(lightwood_config.splitter, lightwood_config)}

log.info('Preparing the encoders')
self.encoders = mut_method_call({{col_name: [encoder, pd.concat(folds[0:nfolds-1])[col_name], 'prepare'] for col_name, encoder in self.encoders.items()}})

log.info('Featurizing the data')
encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)
train_data = encoded_ds_arr[0:nfolds-1]
test_data = encoded_ds_arr[nfolds-1]

log.info('Training the models')
self.models = [{', '.join([call(x, lightwood_config) for x in lightwood_config.output.models])}]
for model in self.models:
    model.fit(train_data)

log.info('Ensembling the model')
self.ensemble = {call(lightwood_config.output.ensemble, lightwood_config)}

log.info('Analyzing the ensemble')
self.model_analysis, self.runtime_analyzer = {call(lightwood_config.analyzer, lightwood_config)}
"""
    learn_body = align(learn_body, 2)

    predict_body = f"""
log.info('Cleaning the data')
data = {call(lightwood_config.cleaner, lightwood_config)}

{ts_code}

encoded_ds = lightwood.encode(self.encoders, data, self.target)
df = self.ensemble(encoded_ds)
insights = {call(lightwood_config.explainer, lightwood_config)}
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
        self.target = '{lightwood_config.output.name}'

    def learn(self, data: pd.DataFrame) -> None:
{learn_body}

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
{predict_body}
"""

    return predictor_code


def generate_predictor(problem_definition: ProblemDefinition = None, data: pd.DataFrame = None, lightwood_config: LightwoodConfig = None) -> str:
    if lightwood_config is None:
        type_information = lightwood.data.infer_types(data, problem_definition.pct_invalid)
        statistical_analysis = lightwood.data.statistical_analysis(data, type_information, problem_definition)
        lightwood_config = lightwood.generate_config(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)

    predictor_code = generate_predictor_code(lightwood_config)
    return predictor_code
