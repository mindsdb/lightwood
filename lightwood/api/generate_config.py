from lightwood.api import LightwoodConfig, TypeInformation, StatisticalAnalysis, Feature, Output
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

    encoder_initialization = encoder_lookup[col_dtype]
    if is_target:
        if col_dtype in target_encoder_lookup_override:
            encoder_initialization = target_encoder_lookup_override[col_dtype]

    encoder_initialization += '('

    # Set arguments for the encoder
    if 'PretrainedLangEncoder' in encoder_initialization and not is_target:
        encoder_initialization += f"""output_type={output.data_dtype}"""

    encoder_initialization += ')'

    return encoder_initialization


def generate_config(target: str, type_information: TypeInformation, statistical_analysis: StatisticalAnalysis) -> LightwoodConfig:

    lightwood_config = LightwoodConfig()
    output = Output()
    output.name = target
    output.data_dtype = type_information.dtypes[target]
    output.encoder = lookup_encoder(type_information.dtypes[target], True, output)
    output.models = '[Nn(), LightGBMMixer()]'
    output.ensemble = 'BestOf'
    lightwood_config.output = output

    for col_name, col_dtype in type_information.dtypes.items():
        if type_information.identifiers[col_name] is None and col_dtype not in (dtype.invalid, dtype.empty) and col_name != target:
            feature = Feature()
            feature.name = col_name
            feature.data_dtype = dtype
            feature.encoder = lookup_encoder(col_dtype, False, output)
            lightwood_config.features[col_name] = feature

    lightwood_config.cleaner = 'cleaner'
    lightwood_config.splitter = 'splitter'
    lightwood_config.analyzer = 'model_analyzer'

    # @TODO: Only import the minimal amount of things we need
    lightwood_config.imports = [
        'from lightwood.model import LightGBMMixer',
        'from lightwood.model import Nn',
        'from lightwood.ensemble import BestOf',
        'from lightwood.data import cleaner',
        'from lightwood.data import splitter'
    ]

    for feature in lightwood_config.features.values():
        encoder_initialization = feature.encoder.split('(')[0]
        lightwood_config.imports.append(f'from lightwood.encoder import {encoder_initialization}')

    lightwood_config.imports = list(set(lightwood_config.imports))
    return lightwood_config
