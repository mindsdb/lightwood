import lightwood
import random
import pandas as pd
import numpy as np
from collections import Counter


random.seed(66)
n = 100
m = 500
train = True

#options = ['a','b','c','d','e','f','g','h','n','m']
options = ['a','b','c']

data_train = {}
data_test = {}

for data, nr_ele in [(data_train,n), (data_test,m)]:
    for i in range(1,6):
        data[f'x_{i}'] = [random.choice(options) for _ in range(nr_ele)]

    data['y'] = [Counter([data[f'x_{i}'][n] for i in range(1,6)]).most_common(1)[0][0] for n in range(nr_ele)]

data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')

if train:
    predictor = lightwood.Predictor(output=['y'])
    predictor.learn(from_data=data_train, callback_on_iter=iter_function, eval_every_x_epochs=200)
    predictor.save('/tmp/ltcrl.pkl')

predictor = lightwood.Predictor(load_from_path='/tmp/ltcrl.pkl')
print('Train accuracy: ', predictor.train_accuracy['y']['value'])
print('Test accuracy: ', predictor.calculate_accuracy(from_data=data_test)['y']['value'])

predictions = predictor.predict(when_data=data_test)
print(f'Confidence mean for all columns present ', np.mean(predictions['y']['selfaware_confidences']))

for i_drop in range(1,6):
    predictions = predictor.predict(when_data=data_test.drop(columns=[f'x_{i_drop}']))
    print(f'Accuracy for x_{i_drop} missing: ', predictor.calculate_accuracy(from_data=data_test.drop(columns=[f'x_{i_drop}']))['y']['value'])
    print(f'Confidence mean for x_{i_drop} missing: ', np.mean(predictions['y']['selfaware_confidences']))
