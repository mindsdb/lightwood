# Intro

Think of Lightwood as Keras for Pytorch. The current implementation is inspired on early work we did in MindsDB and ideas from [Ludwig](https://github.com/uber/ludwig).

The objective is that given a dataset you can do two things:

- Train a model with just one line of code
- Give you full flexibility into changing, modyfing every single part of the process if you so wish to.

The main concept is that ML models can be very modular, and each building block is something that you can thinker with.

The three main building blocks are:

- Encoders: That is, you take some data in and output a vector/tensor representation of this data.
- Mixers: How you mix the output of encoders and also other mixers
- Decoders: How you from a vector/tensor of data to a representation of such data that you wish. (For example: from a vector to actual text)


# Getting started

## Installing Lightwood

```python
pip3 install lightwood
```

You would need python 3.5 or higher.


## How to use

Asume that you have a training file such as this one.

You need to create a configuration dictionary that describes what you want as input and what you want as output as follows:

```python
config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric'
            },
            {
                'name': 'y',
                'type': 'numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'numeric'
            }
        ]
        
    }
```

Then you can ask Ludwig to learn how to predict the output_features:

```python
from lightwood import Predictor
import pandas as pd

predictor = Predictor(definition=config)
predictor.learn(from_data=pd.read_csv(file))
print(predictor.train_accuracy)
print(predictor.predict(when_data=pandas.DataFrame({'x': [1], 'y': [-1]})))

```

## Config 

Configs are made out of three blocks:

### input_features

Its a simple dict that has the following schema

```python
{
    'name': str,
    Optional('type'): any of COLUMN_DATA_TYPES,
    Optional('encoder_class'): object,
    Optional('encoder_attrs'): dict
}
```

* name: is the name of the column as it is in the input data frame
* type: is the type od data contained. Where out of the box, supported COLUMN_DATA_TYPES are:

```python
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    DATETIME = 'datetime'
    IMAGE = 'image'
    TEXT = 'text'
    TIME_SERIES = 'time_series'
```

If you specify the type, it will use the default encoder for that type. 

* encoder_class: This is if you want to replace the default encoder with a different one, so you put the encoder class there
* encoder_attrs: These are the attributes that you want to setup on the encoder once the class its initialized 




