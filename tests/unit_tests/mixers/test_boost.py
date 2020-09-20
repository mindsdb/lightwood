import unittest
import random
import pandas
from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.mixers import BoostMixer


class TestNnMixer(unittest.TestCase):
    def test_fit_and_predict(self):
        pass
        # config = {
        #     'input_features': [
        #         {
        #             'name': 'x',
        #             'type': 'numeric'
        #         },
        #         {
        #             'name': 'y',
        #             'type': 'numeric'
        #         }
        #     ],

        #     'output_features': [
        #         {
        #             'name': 'z',
        #             'type': 'numeric'
        #         },
        #         {
        #             'name': 'z`',
        #             'type': 'categorical'
        #         }
        #     ]
        # }
        # config = predictor_config_schema.validate(config)

        # data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
        # nums = [data['x'][i] * data['y'][i] for i in range(10)]

        # data['z'] = [i + 0.5 for i in range(10)]
        # data['z`'] = ['low' if i < 50 else 'high' for i in nums]

        # data_frame = pandas.DataFrame(data)
        # ds = DataSource(data_frame, config)
        # ds.train()

        # mixer = BoostMixer()
        # mixer.fit(ds, ds)

        # predict_input_ds = DataSource(data_frame[['x', 'y']], config)
        # predict_input_ds._prepare_encoders()
        # predictions = mixer.predict(predict_input_ds)
