from lightwood.api import LightwoodConfig, TypeInformation, StatisticalAnalysis, Feature, Output
from lightwood.api import dtype


def lookup_encoder(col_dtype: dtype, is_target: bool):
    encoder_lookup = {
        dtype.integer: 'NumericEncoder()',
        dtype.float: 'NumericEncoder()',
        dtype.binary: 'OneHotEncoder()',
        dtype.categorical: 'CategoricalAutoEncoder()',
        dtype.tags: 'MultiHotEncoder()',
        dtype.date: 'DatetimeEncoder()',
        dtype.datetime: 'DatetimeEncoder()',
        dtype.image: 'Img2VecEncoder()',
        dtype.rich_text: 'PretrainedLang()',
        dtype.short_text: 'ShortTextEncoder()',
        dtype.array: 'TsRnnEncoder()',
    }

    target_encoder_lookup_override = {
        dtype.rich_text: 'VocabularyEncoder()'
    }

    encoder_class = encoder_lookup[col_dtype]
    if is_target:
        if col_dtype in target_encoder_lookup_override:
            encoder_class = target_encoder_lookup_override[col_dtype]
    return encoder_class

def create_feature(name: str, col_dtype: dtype) -> Feature:
    feature = Feature()
    feature.name = name
    feature.dtype = dtype
    feature.encoder = lookup_encoder(col_dtype, False)
    return feature

def generate_config(target: str, type_information: TypeInformation, statistical_analysis: StatisticalAnalysis) -> LightwoodConfig:

    lightwood_config = LightwoodConfig()
    for col_name, col_dtype in type_information.dtypes.items():
        if type_information.identifiers[col_name] is None and col_dtype not in (dtype.invalid, dtype.empty) and col_name != target:
            lightwood_config.features[col_name] = create_feature(col_name, col_dtype)

    output = Output()
    output.name = target
    output.dtype = type_information.dtypes[target]
    output.encoder = lookup_encoder(type_information.dtypes[target], True)
    output.models = '[Nn(), LightGBM()]'
    output.ensemble = 'BestOf'
    lightwood_config.output = output

    lightwood_config.cleaner = 'cleaner'
    lightwood_config.splitter = 'splitter'
    lightwood_config.analyzer = 'model_analyzer'

    # @TODO: Only import the minimal amount of things we need
    lightwood_config.imports = [
        'from lightwood.model import LightGBM'
        ,'from lightwood.model import Nn'
        ,'from lightwood.ensemble import BestOf'
        ,'from lightwood.data import cleaner'
        ,'from lightwood.data import splitter'
        ,'from lightwood.analysis import model_analyzer'
    ]

    for feature in lightwood_config.features.values():
        encoder_class = feature.encoder.split('(')[0]
        lightwood_config.imports.append(f'from lightwood.encoders import {encoder_class}')

    lightwood_config.imports = list(set(lightwood_config.imports))
    return lightwood_config
