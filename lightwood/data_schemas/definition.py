from schema import Schema, And, Use, Optional
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, HISTOGRAM_TYPES

feature_schema = Schema({
                'name': str,
                'type': And(str, Use(str.lower), lambda s: s in COLUMN_DATA_TYPES.get_attributes().values())
            })

definition_schema = Schema({
        'name': str,
        'input_features': [
            feature_schema
        ],
        'output_features': [
            feature_schema
        ]

    })