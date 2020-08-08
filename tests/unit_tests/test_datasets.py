import unittest
import lightwood
from lightwood import Predictor
import pandas as pd


USE_CUDA = False
PLINEAR = False
CACHE_ENCODED_DATA = False
SELFAWARE = False


class TestDatasets(unittest.TestCase):
    def test_home_rentals(self):
        lightwood.config.config.CONFIG.USE_CUDA = USE_CUDA
        lightwood.config.config.CONFIG.PLINEAR = PLINEAR

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
            'data_source': {'cache_transformed_data': CACHE_ENCODED_DATA},
            'mixer': {'class': lightwood.BUILTIN_MIXERS.NnMixer, 'selfaware': SELFAWARE}
        }

        df=pd.read_csv("https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv")

        def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
            print('epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
                iter=epoch,
                error=error,
                test_error=test_error,
                test_error_gradient=test_error_gradient,
                accuracy=predictor.train_accuracy,
                test_accuracy=test_accuracy
            ))

        predictor = Predictor(config)
        # stop_training_after_seconds given in order to not get timeouts in travis
        predictor.learn(
            from_data=df,
            callback_on_iter=iter_function,
            eval_every_x_epochs=4,
            stop_training_after_seconds=80
        )

        df = df.drop([x['name'] for x in config['output_features']], axis=1)
        predictor.predict(when_data=df)

        predictor.save('test.pkl')
        predictor = Predictor(load_from_path='test.pkl')

        for j in range(100):
            pred = predictor.predict(when={'sqft': round(j * 10)})['number_of_rooms']['predictions'][0]
            assert(isinstance(pred, str) or isinstance(pred, int))
