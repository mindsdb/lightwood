import runpy
import os
import sys
import pandas as pd
from lightwood import Predictor
import lightwood

pdir = '../../lightwood/'
encoders_path = pdir + 'encoders/'
mixers_path = pdir + 'mixers/'
MODULES = [
    f'{encoders_path}categorical/onehot.py',
    f'{encoders_path}datetime/datetime.py',
    f'{encoders_path}categorical/autoencoder.py',
    f'{encoders_path}time_series/ts_fresh_ts.py',
    f'{pdir}api/data_source.py',
    f'{mixers_path}nn/nn.py',
    # Take too long
    # f'{encoders_path}text/distilbert.py',
    # f'./lightwood/mixers/boost/boost.py',

    # Not working for some reason
    #f'{encoders_path}text/rnn.py'.format(encoders_path),
    #'./lightwood/encoders/image/nn.py',
    #'./lightwood/encoders/image/img_2_vec.py',
    #'./lightwood/encoders/numeric/numeric.py',
    #'./lightwood/encoders/time_series/cesium_ts.py',
    #'./lightwood/mixers/sk_learn/sk_learn.py',
    #'./lightwood/encoders/text/infersent.py',
]


def run_tests(modules):
    '''
    Run modules as scripts to execute main function
    '''
    for module in modules:
        runpy.run_path(module, run_name='__main__')


def run_full_test(USE_CUDA, CACHE_ENCODED_DATA, SELFAWARE, PLINEAR):
    '''
    Run full test example with home_rentals dataset
    '''
    lightwood.config.config.CONFIG.USE_CUDA = USE_CUDA
    lightwood.config.config.CONFIG.PLINEAR = PLINEAR

    config = {'input_features': [
                        {'name': 'number_of_bathrooms', 'type': 'numeric'}, {'name': 'sqft', 'type': 'numeric'},
                        {'name': 'location', 'type': 'categorical'}, {'name': 'days_on_market', 'type': 'numeric'},
                        {'name': 'neighborhood', 'type': 'categorical','dropout':0.4},{'name': 'rental_price', 'type': 'numeric'}],
     'output_features': [{'name': 'number_of_rooms', 'type': 'categorical',
                      # 'weights':{
                      #       '0': 0.8,
                      #       '1': 0.6,
                      #       '2': 0.5,
                      #       '3': 0.7,
                      #       '4': 1,
                      # }
    }],
    'data_source': {'cache_transformed_data':CACHE_ENCODED_DATA},
    'mixer':{'class': lightwood.BUILTIN_MIXERS.NnMixer, 'selfaware': SELFAWARE}}


    df=pd.read_csv("https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv")


    def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(
            'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
                iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
                accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))

    predictor = Predictor(config)
    # stop_training_after_seconds given in order to not get timeouts in travis
    predictor.learn(from_data=df, callback_on_iter=iter_function, eval_every_x_epochs=20, stop_training_after_seconds=30)

    predictor.save('test.pkl')

    predictor = Predictor(load_from_path='test.pkl')

    df = df.drop([x['name'] for x in config['output_features']], axis=1)
    predictor.predict(when_data=df)


    predictor.save('test.pkl')
    predictor = Predictor(load_from_path='test.pkl')

    preds = {}
    for j in range(100):
        pred = predictor.predict(when={'sqft': round(j * 10)})['number_of_rooms']['predictions'][0]
        if pred not in preds:
            preds[pred] = 0
        preds[pred] += 1


if __name__ == "__main__":
    run_tests(MODULES)
    for USE_CUDA in [False]:
        for CACHE_ENCODED_DATA in [False, True]:
            run_full_test(USE_CUDA, CACHE_ENCODED_DATA, SELFAWARE, PLINEAR)
