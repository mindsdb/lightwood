<h1 align="center">
	<img width="500" src="https://github.com/mindsdb/mindsdb-docs/blob/master/mindsdb-docs/docs/assets/MindsDBLightwood@3x.png" alt="Lightwood">
	<br>
	<br>
</h1>

![Lightwood Actions workflow](https://github.com/mindsdb/lightwood/workflows/Lightwood%20Actions%20workflow/badge.svg)
![](https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg)
[![PyPI version](https://badge.fury.io/py/lightwood.svg)](https://badge.fury.io/py/lightwood)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightwood)
[![Discourse posts](https://img.shields.io/discourse/posts?server=https%3A%2F%2Fcommunity.mindsdb.com%2F)](https://community.mindsdb.com/)

Lightwood is like Legos for Machine Learning. 

A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with one objective:

- Make it so simple that you can build predictive models with as little as one line of code.


# Documentation
Learn more from the [Lightwood's docs](https://docs.mindsdb.com/lightwood/info/).  

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

Train the model.
```python
import pandas
sensor3_predictor = Predictor(output=['sensor3']).learn(from_data=pandas.read_csv('sensor_data.csv'))
```
You can now predict what *sensor3* value will be.

```python
prediction = sensor3_predictor.predict(when={'sensor1':1, 'sensor2':-1})
```

* You can also try Lightwood in Google Colab: [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg "Lightwood")](https://colab.research.google.com/drive/10W43sur_uj28ROiGuAIF9X46_Xrx1e7K)

## Contributing

Thanks for your interest. There are many ways to contribute to this project. Please, check out our [Contribution guide](https://github.com/mindsdb/lightwood/blob/master/CONTRIBUTING.md).

### Current contributors 

<a href="https://github.com/mindsdb/lightwood/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=mindsdb/lightwood" />
</a>

Made with [contributors-img](https://contributors-img.web.app).

## License ![PyPI - License](https://img.shields.io/pypi/l/lightwood)

* [Lightwood License](https://github.com/mindsdb/lightwood/blob/master/LICENSE)
