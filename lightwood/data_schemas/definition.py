from schema import Schema, And, Use, Optional
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, HISTOGRAM_TYPES

input_feature_schema = Schema({
    'name': str,
    'type': And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values()),
    Optional('encoder_path'): str,
    Optional('encoder_args'): dict
})

output_feature_schema = Schema({
    'name': str,
    'type': And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values())

})

default_mixer_schema = Schema({
    'class': object,
    'argument': dict
})

mixer_schema = Schema({
    'name': str,
    'input': list,
    'output': list,
    'class': object,
    'argument': dict
})

definition_schema = Schema({
    'name': str,
    'input_features': [
        input_feature_schema
    ],
    'output_features': [
        output_feature_schema
    ],
    Optional('default_mixer'): default_mixer_schema,
    Optional('mixers'): [mixer_schema]

})




