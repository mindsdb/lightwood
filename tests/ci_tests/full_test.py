import os
import sys
import pandas as pd
from lightwood import Predictor
import lightwood


def run_test(USE_CUDA, CACHE_ENCODED_DATA, SELFAWARE, PLINEAR):
    lightwood.config.config.CONFIG.USE_CUDA = USE_CUDA
    lightwood.config.config.CONFIG.CACHE_ENCODED_DATA = CACHE_ENCODED_DATA
    lightwood.config.config.CONFIG.SELFAWARE = SELFAWARE
    lightwood.config.config.CONFIG.PLINEAR = PLINEAR

    ####################
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
     'mixer':{'class': lightwood.BUILTIN_MIXERS.NnMixer}}


    # AX doesn't seem to work on the travis version of windows, so don't test it there as of now
    if sys.platform not in ['win32','cygwin','windows']:
        pass
        #config['optimizer'] = lightwood.model_building.BasicAxOptimizer



    df=pd.read_csv("https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv")


    def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
        print(
            'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
                iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
                accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))

    predictor = Predictor(config)
    # stop_training_after_seconds given in order to not get timeouts in travis
    predictor.learn(from_data=df, callback_on_iter=iter_function, eval_every_x_epochs=1, stop_training_after_seconds=1)

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

for USE_CUDA in [False]:
    for CACHE_ENCODED_DATA in [False, True]:
        for SELFAWARE in [False, True]:
            for PLINEAR in [False, True]:
                run_test(USE_CUDA, CACHE_ENCODED_DATA, SELFAWARE, PLINEAR)
