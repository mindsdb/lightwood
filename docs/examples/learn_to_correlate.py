import lightwood
import random
import pandas as pd
import numpy as np

lightwood.config.config.CONFIG.HELPER_MIXERS = False
random.seed(66)

n = 100
m = 11
train = False

data_train = {}
data_test = {}

for data, nr_ele in [(data_train,n), (data_test,m)]:
    for i in range(1,5):
        data[f'x_{i}'] = [random.randint(25,50) for _ in range(nr_ele)]

    data['y'] = [data['x_1'][i] * 0.9 + data['x_2'][i] * 0.09 + data['x_3'][i] * 0.009 + data['x_4'][i] * 0.0009 for i in range(nr_ele)]

data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)



def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')

if train:
    predictor = lightwood.Predictor(output=['y'])
    predictor.learn(from_data=data_train, callback_on_iter=iter_function, eval_every_x_epochs=200)
    predictor.save('/tmp/ltcrl.pkl')

predictor = lightwood.Predictor(load_from_path='/tmp/ltcrl.pkl')
print('Train accuracy: ', predictor.train_accuracy)
print('Test accuracy: ', predictor.calculate_accuracy(from_data=data_test))

for i_drop in range(1,5):
    predictions = predictor.predict(when_data=data_test.drop(columns=[f'x_{i_drop}']))
    print(f'Accuracy for x_{i_drop} missing: ', predictor.calculate_accuracy(from_data=data_test.drop(columns=[f'x_{i_drop}'])))
    print(f'Confidence mean for x_{i_drop} missing: ', np.mean(predictions['y']['selfaware_confidences']))
