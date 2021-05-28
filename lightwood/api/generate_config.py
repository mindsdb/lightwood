from typing import Dict
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


def generate_config(type_information: TypeInformation, statistical_analysis: StatisticalAnalysis, problem_definition: ProblemDefinition) -> LightwoodConfig:
    
    target = problem_definition.target

    output = Output(
        name=target,
        data_dtype=type_information.dtypes[target],
        encoder=None,
        models='[Nn(self.lightwood_config), LightGBM(self.lightwood_config)]',
        ensemble='BestOf'
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
        'from lightwood.model import Nn',
        'from lightwood.ensemble import BestOf',
        'from lightwood.data import cleaner',
        'from lightwood.data import splitter'
    ]

    for feature in features.values():
        encoder_initialization = feature.encoder.split('(')[0]
        imports.append(f'from lightwood.encoder import {encoder_initialization}')

    imports = list(set(imports))
    return LightwoodConfig(
        cleaner='cleaner',
        splitter='splitter',
        analyzer='model_analyzer',
        features=features,
        output=output,
        imports=imports,
        problem_definition=problem_definition
    )
