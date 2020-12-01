from schema import Schema, And, Use, Optional
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import NnMixer

feature_schema = Schema({
    'name': str,
    'type': And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values()),
    Optional('encoder_class'): object,
    Optional('encoder_attrs'): dict,
    Optional('depends_on_column'): list,
    Optional('dropout'): float,
    Optional('weights'): dict,
    Optional('secondary_type'): And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values()),
    Optional('original_type'): And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values()),
    Optional('additional_info'): dict
})

mixer_schema = Schema({
    Optional('class', default=NnMixer): object,
    Optional('kwargs', default={}): dict
})

data_source_schema = Schema({
    Optional('cache_transformed_data', default=True): bool,
})

predictor_config_schema = Schema({
    'input_features': [
        feature_schema
    ],
    'output_features': [
        feature_schema
    ],
    Optional('data_source', default=data_source_schema.validate({})): data_source_schema,
    Optional('mixer', default=mixer_schema.validate({'class': NnMixer, 'kwargs': {}})): mixer_schema,
    Optional('optimizer'): object
})
