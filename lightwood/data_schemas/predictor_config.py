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
    'class': object,
    Optional('attrs'): dict,
    Optional('mixer_graph'): [mixer_graph_schema]
})

data_source_schema = Schema({
    Optional('cache_transformed_data', default=True): bool,
})

default_data_source_config = {
    'cache_transformed_data': True
}

predictor_config_schema = Schema({
    'input_features': [
        feature_schema
    ],
    'output_features': [
        feature_schema
    ],
    Optional('data_source', default=default_data_source_config): data_source_schema,
    Optional('mixer'): mixer_schema,
    Optional('optimizer'): object
})
