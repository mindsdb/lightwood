import lightwood
import pandas as pd
import numpy as np
import random
from collections import Counter


random.seed(66)

def gen_multiply():
    n = 1000

    # generate random numbers between -10 and 10
    data_train = {'x': [random.randint(-15, 5) for i in range(n)],
            'y': [random.randint(-15, 5) for i in range(n)]}

    data_test = {'x': [random.randint(-5, 15) for i in range(n)],
            'y': [random.randint(-5, 15) for i in range(n)]}

    # target variable to be the multiplication of the two
    data_train['z'] = [data_train['x'][i] * data_train['y'][i] for i in range(n)]
    data_test['z'] = [data_test['x'][i] * data_test['y'][i] for i in range(n)]
    
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    return (df_train, df_test, [['x'],['y']], 'z','multiplied')

def gen_correlate():
    n = 500
    train = True

    data_train = {}
    data_test = {}

    for data, nr_ele in [(data_train,n), (data_test,n)]:
        for i in range(1,5):
            data[f'x_{i}'] = [random.random()*50 + 25  for _ in range(nr_ele)]

        data['y'] = [data['x_1'][i] * 0.9 + data['x_2'][i] * 0.09 + data['x_3'][i] * 0.009 + data['x_4'][i] * 0.0009 for i in range(nr_ele)]

    data_train = pd.DataFrame(data_train)
    data_test = pd.DataFrame(data_test)
    return (data_train, data_test, [['x_1'],['x_2','x_4'],['x_3'],['x_2']], 'y','correlated')

def gen_categorical():
    n = 1000
    nr_inputs = 6

    options = ['a','b','c']

    data_train = {}
    data_test = {}

    for data, nr_ele in [(data_train,n), (data_test,int(n/5))]:
        for i in range(nr_inputs):
            data[f'x_{i}'] = [random.choice(options) for _ in range(nr_ele)]

        data['y'] = [Counter([data[f'x_{i}'][n] for i in range(nr_inputs)]).most_common(1)[0][0] for n in range(nr_ele)]

    data_train = pd.DataFrame(data_train)
    data_test = pd.DataFrame(data_test)
    return (data_train, data_test, [['x_1'],['x_2','x_4'],['x_5','x_4','x_3']], 'y','cummulative')


def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')


test_cases = [gen_multiply(),gen_correlate(),gen_categorical()]

log_map = {}
for i, data in enumerate(test_cases):
    df_train, df_test, dropout_arr, out_col, name = data

    pmap = {}
    accmap = {}

    pmap['normal'] = lightwood.Predictor(output=[out_col])
    pmap['normal'].learn(from_data=df_train, callback_on_iter=iter_function, eval_every_x_epochs=100)
    accmap['normal'] = pmap['normal'].calculate_accuracy(from_data=df_test)[out_col]['value']

    for cols in dropout_arr:
        mk = 'missing_' + '_'.join(cols)
        pmap[mk] = lightwood.Predictor(output=[out_col])
        pmap[mk].learn(from_data=df_train.drop(columns=cols), callback_on_iter=iter_function, eval_every_x_epochs=100)
        accmap[mk + '_unfit'] = pmap['normal'].calculate_accuracy(from_data=df_test.drop(columns=cols))[out_col]['value']
        accmap[mk + '_fit'] = pmap[mk].calculate_accuracy(from_data=df_test.drop(columns=cols))[out_col]['value']

    text = f'\n---------\nTest case {name}\n---------\nNormal accuracy of: ' + str(accmap['normal'])
    for cols in dropout_arr:
        mk = 'missing_' + '_'.join(cols)
        text += f'\nSpecially-trained trained accuracy when {cols} missing: ' + str(accmap[mk + '_fit'])
        text += f'\nNormally-trained trained accuracy when {cols} missing: ' + str(accmap[mk + '_unfit'])

    log_map[name] = text

for k in log_map:
    print(log_map[k])
