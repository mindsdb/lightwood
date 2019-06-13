

Think of Lightwood as [Keras](https://keras.io/)+[Ludwig](https://github.com/uber/ludwig) but for Pytorch, with two objectives:

- Make it so simple that you can build models with just a few lines of code.
- Make it so flexible that you can change and customize everything.



## Installing Lightwood

```bash
pip3 install lightwood
```

You would need python 3.5 or higher.


## Quick example

Asume that you have a training file (sensor_data.csv) such as this one. 

| sensor1  | sensor2 | sensor3 |
|----|----|----|
|  1 | -1 | -1 |
| 0  | 1  | 0  |
| -1 | -1 | 1  |
| 1  | 0  | 0  |
| 0  | 1  | 0  |
| -1 | 1  | -1 |
| 0  | 0  | 0  |
| -1 | -1 | 1  |
| 1  | 0  | 0  |

And you would like to learn to predict the values of *sensor3* given the readings in *sensor1* and *sensor2*.

### Learn

You can train a Predictor as follows:

```python
from lightwood import Predictor
import pandas

sensor3_predictor = Predictor(output=['sensor3'])
sensor3_predictor.learn(from_data=pandas.read_csv('sensor_data.csv'))

```

### Predict 

You can now given new readings from *sensor1* and *sensor2* predict what *sensor3* will be.

```python

prediction = sensor3_predictor.predict(when={'sensor1':1, 'sensor2':-1})
print(prediction)
```

Of course that example was just the tip of the iceberg, please read about the main concepts of lightwood, [the API](API.md) and then jump into examples.
