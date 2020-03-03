
![Lightwood](https://mindsdb.github.io/lightwood/assets/logo.png)
#

[![Build Status](https://travis-ci.org/mindsdb/lightwood.svg?branch=master)](https://travis-ci.org/mindsdb/lightwood)
[![PyPI version](https://badge.fury.io/py/lightwood.svg)](https://badge.fury.io/py/lightwood)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightwood)

Lightwood is like Legos for Machine Learning. 

A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with on objective:

- Make it so simple that you can build predictive models with as little as one line of code.


# Documentation
Learn more  from the [Lightwood's docs](https://mindsdb.github.io/lightwood/API/).  

# Quick start
```python
pip3 install lightwood
```

### Learn

You can train a Predictor as follows:

```python
from lightwood import Predictor
sensor3_predictor = Predictor(output=['sensor3']).learn(from_data=pandas.read_csv('sensor_data.csv'))

```

### Predict

You can now given new readings from *sensor1* and *sensor2* predict what *sensor3* will be.

```python

prediction = sensor3_predictor.predict(when={'sensor1':1, 'sensor2':-1})

```
