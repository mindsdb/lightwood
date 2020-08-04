from copy import copy

from unittest import TestCase
import random
import numpy as np
import pandas as pd
from torch import Tensor

from lightwood.api.data_source import DataSource

from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.encoders import NumericEncoder, CategoricalAutoEncoder


class TestDataSource(TestCase):
    def setUp(self):
        config = {
            'input_features': [
                {
                    'name': 'x1',
                    'type': 'numeric',

                },
                {
                    'name': 'x2',
                    'type': 'numeric',

                }
            ],

            'output_features': [
                {
                    'name': 'y',
                    'type': 'categorical',

                }
            ]
        }
        config = predictor_config_schema.validate(config)
        n_points = 100
        data = {'x1': [i for i in range(n_points)],
                'x2': [random.randint(i, i + 20) for i in range(n_points)]}
        nums = [data['x1'][i] * data['x2'][i] for i in range(n_points)]

        data['y'] = ['low' if i < 50 else 'high' for i in nums]

        df = pd.DataFrame(data)

        self.config = config
        self.df = df

    def test_prepare_encoders(self):
        df, config = self.df, self.config
        ds = DataSource(df, config)
        assert not ds.disable_cache
        ds.prepare_encoders()

        encoders = ds.encoders

        for col in ['x1', 'x2']:
            assert isinstance(encoders[col], NumericEncoder)
            assert encoders[col]._prepared is True
            assert encoders[col].is_target is False
            assert encoders[col]._type == 'int'

        assert isinstance(encoders['y'], CategoricalAutoEncoder)
        assert encoders['y']._prepared is True
        assert encoders['y'].is_target is True
        assert encoders['y'].onehot_encoder._prepared is True
        assert encoders['y'].onehot_encoder.is_target is True
        assert encoders['y'].use_autoencoder is False

        encoded_column_x1 = ds.get_encoded_column_data('x1')
        assert isinstance(encoded_column_x1, Tensor)
        assert encoded_column_x1.shape[0] == len(df)
        encoded_column_x2 = ds.get_encoded_column_data('x2')
        assert isinstance(encoded_column_x2, Tensor)
        assert encoded_column_x2.shape[0] == len(df)
        encoded_column_y = ds.get_encoded_column_data('y')
        assert isinstance(encoded_column_y, Tensor)
        assert encoded_column_y.shape[0] == len(df)

    def test_encoded_cache(self):
        df, config = self.df, self.config

        ds = DataSource(df, config)
        assert not ds.disable_cache
        ds.prepare_encoders()

        for column in ['x1', 'x2', 'y']:
            assert not column in ds.encoded_cache
            encoded_column = ds.get_encoded_column_data(column)
            assert (ds.encoded_cache[column] == encoded_column).all()

    def test_transformed_cache(self):
        df, config = self.df, self.config

        ds = DataSource(df, config)
        assert ds.disable_cache is False
        ds.prepare_encoders()

        assert ds.transformed_cache is None
        encoded_row = ds[0] # This creates ds.transformed_cache
        assert len(ds.transformed_cache) == len(df)
        assert ds.transformed_cache[0] == encoded_row

        for i in range(1, len(df)):
            assert ds.transformed_cache[i] is None
            encoded_row = ds[i]
            assert ds.transformed_cache[i] == encoded_row

        alternate_config = copy(config)
        alternate_config['data_source']['cache_transformed_data'] = False
        ds = DataSource(df, alternate_config)
        assert ds.disable_cache is True
        ds.prepare_encoders()

        for i in range(len(df)):
            encoded_row = ds[i]
            assert ds.transformed_cache is None
