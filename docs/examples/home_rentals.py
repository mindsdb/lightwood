import pandas as pd
from lightwood import Predictor
import numpy

####################
config_nr = {'input_features': [
                    {'name': 'number_of_bathrooms', 'type': 'numeric'}, {'name': 'sqft', 'type': 'numeric'},
                    {'name': 'location', 'type': 'categorical'}, {'name': 'days_on_market', 'type': 'numeric'},
                    {'name': 'neighborhood', 'type': 'categorical'}],
 'output_features': [{'name': 'number_of_rooms', 'type': 'categorical'}]}


config_rp = {'input_features': [ {'name': 'number_of_rooms', 'type': 'numeric'},
                    {'name': 'number_of_bathrooms', 'type': 'numeric'}, {'name': 'sqft', 'type': 'numeric'},
                    {'name': 'location', 'type': 'categorical'}, {'name': 'days_on_market', 'type': 'numeric'},
                    {'name': 'neighborhood', 'type': 'categorical'}],
 'output_features': [{'name': 'rental_price', 'type': 'numeric'}]}



df=pd.read_csv("https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/home_rentals.csv")

predictor = None
file = '/tmp/predictor.mdb'

def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
    print(
        'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, accuracy: {accuracy}, test_accuracy: {test_accuracy}'.format(
            iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
            accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))

config = config_rp

# predictor = Predictor(config)
# predictor.learn(from_data=df, callback_on_iter=iter_function, eval_every_x_epochs=10)
# predictor.save('/tmp/predictor.mdb')

predictor = Predictor(load_from_path=file)

print(predictor.overall_certainty)
ret = []
for i in range(int(100-predictor.overall_certainty*100)):
    ret_val = predictor.predict(when={'number_of_bedrooms': 2, 'sqft': 2300})[
                config['output_features'][0]['name']]['predictions'][0]
    if ret_val is not None:
        ret += [ret_val]



hist, bin_edges = numpy.histogram(ret)
net = sum(hist)

for j, count in enumerate(hist):
    print('{edge:.2f}: {value:.2%}'.format(value=count/net, edge=bin_edges[j]))
