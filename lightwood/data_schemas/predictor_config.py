from schema import Schema, And, Use, Optional
from lightwood.constants.lightwood import COLUMN_DATA_TYPES

feature_schema = Schema({
    'name': str,
    'type': And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values()),
    Optional('encoder_class'): object,
    Optional('encoder_attrs'): dict,
    Optional('depends_on_column'): str,
    Optional('dropout'): float,
    Optional('weights'): dict
})

mixer_graph_schema = Schema({
    'name': str,
    'input': list,
    Optional('output'): list,
    'class': object,
    Optional('attrs'): dict
})

mixer_schema = Schema({
    Optional('class'): object,
    Optional('attrs'): dict,
    Optional('deterministic', default=True): bool,
    Optional('selfaware', default=True): bool
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
    Optional('mixer', default=mixer_schema.validate({})): mixer_schema,
    Optional('optimizer'): object
})
