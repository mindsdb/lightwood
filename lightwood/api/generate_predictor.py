from typing import Dict

from nltk.corpus.reader import dependency
from lightwood.api.types import ProblemDefinition
import lightwood
import pprint
from lightwood.api import LightwoodConfig
from mindsdb_datasources import DataSource


def dump_config(lightwood_config: LightwoodConfig) -> str:
    config_dump: Dict[str, object] = lightwood_config.to_dict()
    del config_dump['analyzer']
    del config_dump['cleaner']
    del config_dump['splitter']
    for feature in config_dump['features'].values():
        del feature['encoder']
        del feature['dependency']
    del config_dump['imports']
    del config_dump['output']['encoder']
    del config_dump['output']['models']
    del config_dump['output']['ensemble']
    return pprint.pformat(config_dump)


def generate_predictor_code(lightwood_config: LightwoodConfig) -> str:
    feature_code_arr = []
    dependency_arr = []
    for feature in lightwood_config.features.values():
        feature_code_arr.append(f"""'{feature.name}': {feature.encoder}""")
        dependency_arr.append(f"""'{feature.name}': {feature.dependency}""")

    encoder_code = '{\n            ' + ',\n            '.join(feature_code_arr) + '\n        }'
    dependency_code = '{\n            ' + ',\n            '.join(dependency_arr) + '\n        }'

    import_code = '\n'.join(lightwood_config.imports)
    config_dump = dump_config(lightwood_config)

    return f"""{import_code}
import pandas as pd
from mindsdb_datasources import DataSource
from lightwood.helpers.seed import seed
from lightwood.helpers.log import log
import lightwood
from lightwood.api import LightwoodConfig
from lightwood.model import BaseModel
from lightwood.encoder import BaseEncoder
from lightwood.ensemble import BaseEnsemble
from typing import Dict, List


class Predictor():
    target: str
    lightwood_config: LightwoodConfig
    models: List[BaseModel]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble

    def __init__(self):
        seed()
        self.target = '{lightwood_config.output.name}'

    def learn(self, data: DataSource) -> None:
        # Build a Graph from the JSON
        # Using eval is a bit ugly and we could replace it with factories, personally I'm against this, as it ads pointless complexity
        self.lightwood_config = LightwoodConfig.from_dict({config_dump})
        self.encoders = {encoder_code}
        self.dependencies = {dependency_code}

        log.info('Cleaning up, transforming and splitting the data')
        data = {lightwood_config.cleaner}(data, self.lightwood_config)
        folds = {lightwood_config.splitter}(data, 10)
        nfolds = len(folds)

        log.info('Training the encoders')
        for col_name, encoder in self.encoders.items():
            # @TODO recursive later to handle depndency columns that have dependencies
            if len(self.dependencies[col_name]) > 0:
                for dep_col in self.dependencies[col_name]:
                    log.info('Preparting encoder for column: ' + col_name)
                    if encoder.uses_folds:
                        encoder.prepare([x[dep_col] for x in folds[0:nfolds-1]])
                    else:
                        encoder.prepare(pd.concat(folds[0:nfolds-1])[dep_col])  
            
            if not encoder._prepared:
                log.info('Preparting encoder for column: ' + col_name)
                if encoder.uses_folds:
                    encoder.prepare([x[col_name] for x in folds[0:nfolds-1]])
                else:
                    encoder.prepare(pd.concat(folds[0:nfolds-1])[col_name])    

        log.info('Featurizing the data')
        encoded_ds_arr = lightwood.encode(self.encoders, folds, self.target)

        log.info('Training the models')
        self.models = {lightwood_config.output.models}
        for model in self.models:
            model.fit(encoded_ds_arr[0:nfolds-1])

        log.info('Ensembling the model')
        self.ensemble = {lightwood_config.output.ensemble}(self.models, encoded_ds_arr[nfolds-1], self.lightwood_config)

        log.info('Analyzing the ensemble')
        # Add back when analysis works
        # self.confidence_model, self.predictor_analysis = {lightwood_config.analyzer}(self.ensemble, encoded_ds_arr[nfolds-1], data[nfolds-1])

    def predict(self, data: DataSource) -> pd.DataFrame:
        encoded_ds_arr = lightwood.encode(self.encoders, data)
        df = self.ensemble(encoded_ds_arr)
        return df

"""


def generate_predictor(problem_definition: ProblemDefinition = None, datasource: DataSource = None, lightwood_config: LightwoodConfig = None) -> str:
    if lightwood_config is None:
        type_information = lightwood.data.infer_types(datasource, problem_definition.pct_invalid)
        statistical_analysis = lightwood.data.statistical_analysis(datasource, type_information)
        lightwood_config = lightwood.generate_config(type_information=type_information, statistical_analysis=statistical_analysis, problem_definition=problem_definition)

    predictor_code = generate_predictor_code(lightwood_config)
    return predictor_code
