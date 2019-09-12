from schema import Schema, And, Use, Optional
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, HISTOGRAM_TYPES

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



predictor_config_schema = Schema({
    'input_features': [
        feature_schema
    ],
    'output_features': [
        feature_schema
    ],
    Optional('mixer'): mixer_schema,
    Optional('optimizer'): object
})
