import requests
import os
import tarfile
import pandas as pd
from lightwood import Predictor


with open(f'cifar_100.tar.gz', 'wb') as f:
    r = requests.get(f'https://mindsdb-example-data.s3.eu-west-2.amazonaws.com/cifar_100.tar.gz')
    f.write(r.content)

try:
    tar = tarfile.open(f'cifar_100.tar.gz', 'r:gz')
except:
    tar = tarfile.open(f'cifar_100.tar.gz', 'r')

tar.extractall()
tar.close()

os.chdir('cifar_100')

config = {'input_features': [{'name': 'image_path', 'type': 'image', 'encoder_attrs': {'aim': 'speed'}}], 'output_features': [{'name': 'superclass', 'type': 'categorical', 'encoder_attrs': {}}, {'name': 'class', 'type': 'categorical', 'encoder_attrs': {}}]}
predictor = Predictor(config)


def iter_function(epoch, error, test_error, test_error_gradient):
    print(
        'epoch: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}, accuracy: {accuracy}'.format(
            iter=epoch, error=error, test_error=test_error, test_error_gradient=test_error_gradient,
            accuracy=predictor.train_accuracy))


predictor.learn(from_data=pd.read_csv('train_sample.csv'), callback_on_iter=iter_function, eval_every_x_epochs=1)

results = predictor.predict(when_data=pd.read_csv('test_sample.csv'))
print(results)
