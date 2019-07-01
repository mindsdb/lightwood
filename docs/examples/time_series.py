import math
import pandas
import random
from lightwood import Predictor
from lightwood import COLUMN_DATA_TYPES



ts_len = 10
max = 2000
total_rows = 100
ts_data = [
    [
        " ".join([str(i/10) for i in range(j+1, j+ts_len)]),
        " ".join([str(math.sin(i/max)) for i in range(j+1, j+ts_len)]),
        math.sin((j+ts_len+1) / max)
    ]  for j in range(total_rows)
]

config = {'input_features': [{'name': 'ts', 'type': COLUMN_DATA_TYPES.TIME_SERIES }],
 'output_features': [{'name': 'next', 'type': 'numeric'}]}



def iter_function(epoch, error, test_error, test_error_gradient):
    print(
        'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, accuracy: {accuracy}'.format(
            iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
            accuracy=predictor.train_accuracy))


data = pandas.DataFrame(ts_data, columns=['time', 'ts', 'next'])

predictor = Predictor(config)


predictor.learn(from_data=data, callback_on_iter=iter_function, eval_every_x_epochs=10)


ret = predictor.predict(when={'ts':" ".join([str(math.sin(i/max)) for i in range(10+1, 10+ts_len)])})
print(" ".join([str(math.sin(i/max)) for i in range(10+1, 10+ts_len+1)]))
print(ret)