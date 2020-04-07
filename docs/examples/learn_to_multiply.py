import pandas
import random
import lightwood
from lightwood import Predictor
import os
from sklearn.metrics import r2_score
import numpy as np


lightwood.config.config.CONFIG.HELPER_MIXERS = False
lightwood.config.config.CONFIG.ENABLE_DROPOUT = False
random.seed(66)
### Generate a dataset
n = 100
m = n * 100
op = '*'

# generate random numbers between -10 and 10
data_train = {'x': [random.randint(-15, 5) for i in range(n)],
        'y': [random.randint(-15, 5) for i in range(n)]}

data_test = {'x': [random.randint(-15, 5) for i in range(m)],
        'y': [random.randint(-15, 5) for i in range(m)]}

if op == '/':
    for i in range(n):
        if data_train['y'][i] == 0:
            data_train['y'][i] = 1
if op == '/':
    for i in range(m):
        if data_test['y'][i] == 0:
            data_test['y'][i] = 1

# target variable to be the multiplication of the two
data_train['z'] = eval(f"""[data_train['x'][i] {op} data_train['y'][i] for i in range(n)]""")
data_test['z'] = eval(f"""[data_test['x'][i] {op} data_test['y'][i] for i in range(m)]""")


df_train = pandas.DataFrame(data_train)
df_test = pandas.DataFrame(data_test)

predictor = Predictor(output=['z'])

def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')

predictor.learn(from_data=df_train, callback_on_iter=iter_function, eval_every_x_epochs=200)
predictor.save('ok.pkl')

predictor = Predictor(load_from_path='ok.pkl')
print('Train accuracy: ', predictor.train_accuracy)
print('Test accuracy: ', predictor.calculate_accuracy(from_data=df_test))

predictions = predictor.predict(when_data=df_test)
print('Confidence mean for both x and y present: ', np.mean(predictions['z']['selfaware_confidences']))
print(list(df_test['z'])[30:60])
print(predictions['z']['predictions'][30:60])

predictions = predictor.predict(when_data=df_test.drop(columns=['x']))
print('Confidence mean for x missing: ', np.mean(predictions['z']['selfaware_confidences']))

predictions = predictor.predict(when_data=df_test.drop(columns=['y']))
print('Confidence mean for y missing: ', np.mean(predictions['z']['selfaware_confidences']))
