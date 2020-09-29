import unittest
import random
import pandas
from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.mixers import NnMixer


class TestNnMixer(unittest.TestCase):
    def test_fit_and_predict(self):
        config = {
            'input_features': [
                {
                    'name': 'x',
                    'type': 'numeric'
                },
                {
                    'name': 'y',
                    'type': 'numeric'
                }
            ],

            'output_features': [
                {
                    'name': 'z',
                    'type': 'numeric'
                },
                {
                    'name': 'z`',
                    'type': 'categorical'
                }
            ]
        }
        config = predictor_config_schema.validate(config)

        N = 100

        data = {'x': [i for i in range(N)], 'y': [random.randint(i, i + 20) for i in range(N)]}
        nums = [data['x'][i] * data['y'][i] for i in range(N)]

        data['z'] = [i + 0.5 for i in range(N)]
        data['z`'] = ['low' if i < 50 else 'high' for i in nums]

        data_frame = pandas.DataFrame(data)
        train_ds = DataSource(data_frame, config)
        train_ds.create_subsets(1)

        mixer = NnMixer(stop_training_after_seconds=50)
        mixer.fit(train_ds, train_ds)

        test_ds = train_ds.make_child(data_frame[['x', 'y']])
        predictions = mixer.predict(test_ds)
