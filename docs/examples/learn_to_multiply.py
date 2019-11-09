import pandas
import random
from lightwood import Predictor
import os

### Generate a dataset
datapoints = 1000

# generate random numbers between -10 and 10
data = {'x': [random.randint(-10, 10) for i in range(datapoints)],
        'y': [random.randint(-10, 10) for i in range(datapoints)]}

# target variable to be the multiplication of the two
data['z'] = [data['x'][i] * data['y'][i] for i in range(datapoints)]


data_frame = pandas.DataFrame(data)
print(data_frame)

predictor = Predictor(output=['z'])


def iter_function(epoch, error, test_error, test_error_gradient, test_accuracy):
    print(
        'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, test_accuracy: {test_accuracy}'.format(
            iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
            accuracy=predictor.train_accuracy, test_accuracy=test_accuracy))



predictor.learn(from_data=data_frame, callback_on_iter=iter_function)
print('accuracy')
print(predictor.train_accuracy)
print('accuracy over all dataset')
print(predictor.calculate_accuracy(from_data=data_frame))
when = {'x': [1], 'y': [0]}
print('- multiply when. {when}'.format(when=when))
print(predictor.predict(when=when))

# saving the predictor
predictor.save('ok.pkl')

# loading the predictor

predictor2 = Predictor(load_from_path='ok.pkl')
when = {'x': [0, 0, 1, -1, 1], 'y': [0, 1, -1, -1, 1]}
print('- multiply when. {when}'.format(when=when))
print(predictor2.predict(when_data=pandas.DataFrame(when)))
when = {'x': [0, 3, 1, -5, 1], 'y': [0, 1, -5, -4, 7]}
print('- multiply when. {when}'.format(when=when))
print(predictor2.predict(when_data=pandas.DataFrame(when)))
