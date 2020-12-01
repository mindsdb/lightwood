import unittest
import pandas as pd
from lightwood import Predictor
from lightwood.mixers import NnMixer, BoostMixer, SklearnMixer


class TestPredictor(unittest.TestCase):
    def test_learn_and_predict_nnmixer(self):
        config = {
            'input_features': [
                {'name': 'sqft', 'type': 'numeric'},
                {'name': 'days_on_market', 'type': 'numeric'},
                {'name': 'neighborhood', 'type': 'categorical', 'dropout': 0.4}
            ],
            'output_features': [
                {'name': 'number_of_rooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 0.5, '3': 0.7, '4': 1}},
                {'name': 'number_of_bathrooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 4}},
                {'name': 'rental_price', 'type': 'numeric'},
                {'name': 'location', 'type': 'categorical'}
            ],
            'mixer': {
                'class': NnMixer,
                'kwargs': {
                    'eval_every_x_epochs': 4,
                    'stop_training_after_seconds': 10
                }
            }
        }

        df = pd.read_csv('https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv')

        predictor = Predictor(config)
        predictor.learn(from_data=df)

        df = df.drop([x['name'] for x in config['output_features']], axis=1)
        predictor.predict(when_data=df)

    def test_learn_and_predict_boostmixer(self):
        config = {
            'input_features': [
                {'name': 'sqft', 'type': 'numeric'},
                {'name': 'days_on_market', 'type': 'numeric'},
                {'name': 'neighborhood', 'type': 'categorical', 'dropout': 0.4}
            ],
            'output_features': [
                {'name': 'number_of_rooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 0.5, '3': 0.7, '4': 1}},
                {'name': 'number_of_bathrooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 4}},
                {'name': 'rental_price', 'type': 'numeric'},
                {'name': 'location', 'type': 'categorical'}
            ],
            'mixer': {'class': BoostMixer}
        }

        df = pd.read_csv('https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv')

        predictor = Predictor(config)
        predictor.learn(from_data=df)

        df = df.drop([x['name'] for x in config['output_features']], axis=1)
        predictor.predict(when_data=df)

    def test_learn_and_predict_sklearn_mixer(self):
        config = {
            'input_features': [
                {'name': 'sqft', 'type': 'numeric'},
                {'name': 'days_on_market', 'type': 'numeric'},
                {'name': 'neighborhood', 'type': 'categorical', 'dropout': 0.4}
            ],
            'output_features': [
                {'name': 'number_of_rooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 0.5, '3': 0.7, '4': 1}},
                {'name': 'number_of_bathrooms', 'type': 'categorical', 'weights': {'0': 0.8, '1': 0.6, '2': 4}},
                {'name': 'rental_price', 'type': 'numeric'},
                {'name': 'location', 'type': 'categorical'}
            ],
            'mixer': {'class': SklearnMixer}
        }

        df = pd.read_csv('https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv')

        predictor = Predictor(config)
        predictor.learn(from_data=df)

        df = df.drop([x['name'] for x in config['output_features']], axis=1)
        predictor.predict(when_data=df)
