
![Lightwood](https://mindsdb.github.io/lightwood/assets/logo.png)
#

[![Build Status](https://travis-ci.org/mindsdb/lightwood.svg?branch=master)](https://travis-ci.org/mindsdb/lightwood)
[![PyPI version](https://badge.fury.io/py/lightwood.svg)](https://badge.fury.io/py/lightwood)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightwood)

Lightwood is like Legos for Machine Learning. 

A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with one objective:

- Make it so simple that you can build predictive models with as little as one line of code.


# Documentation
Learn more from the [Lightwood's docs](https://mindsdb.github.io/lightwood/API/).  

## Try it out

### Installation
You can install Lightwood from pip:

```python
pip3 install lightwood
```
>Note: depending on your environment, you might have to use pip instead of pip3 in the above command.

### Usage
Given the simple sensor_data.csv let's predict sensor3 values.
| sensor1  | sensor2 | sensor3 |
|----|----|----|
|  1 | -1 | -1 |
| 0  | 1  | 0  |
| -1  |- 1  |1  |

Import [Predictor](https://mindsdb.github.io/lightwood/API/) from Lightwood
```python
from lightwood import Predictor
```

Train the model:
```python
import pandas
sensor3_predictor = Predictor(output=['sensor3']).learn(from_data=pandas.read_csv('sensor_data.csv'))
```
You can now predict what *sensor3* value will be.

```python
prediction = sensor3_predictor.predict(when={'sensor1':1, 'sensor2':-1})
```

## License ![PyPI - License](https://img.shields.io/pypi/l/lightwood)

* [Lightwood License](https://github.com/mindsdb/lightwood/blob/master/LICENSE)
