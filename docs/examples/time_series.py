import math
import pandas
import lightwood
from lightwood import Predictor
from lightwood import COLUMN_DATA_TYPES

lightwood.config.config.CONFIG.USE_CUDA = True

ts_len = 100
max = 2000
total_rows = 100
ts_data = [
    [
        [i/10 for i in range(j+1, j+ts_len)],
        [math.sin(i/max) for i in range(j+1, j+ts_len)],
        math.sin((j+ts_len+1) / max)
    ]  for j in range(total_rows)
]

config = {'input_features': [{'name': 'ts', 'type': COLUMN_DATA_TYPES.TIME_SERIES }],
 'output_features': [{'name': 'next', 'type': 'numeric'}]}


def iter_function(epoch, training_error, test_error, test_error_gradient, test_accuracy):
    print(f'Epoch: {epoch}, Train Error: {training_error}, Test Error: {test_error}, Test Error Gradient: {test_error_gradient}, Test Accuracy: {test_accuracy}')


data = pandas.DataFrame(ts_data, columns=['time', 'ts', 'next'])
predictor = Predictor(config)

predictor.learn(from_data=data)

print('\n\n')
ret = predictor.predict(when={'ts':[math.sin(i/max) for i in range(10+1, 10+ts_len)]})
print([math.sin(i/max) for i in range(10+1, 10+ts_len+1)])
print('Got predictions: ')
print(ret)
