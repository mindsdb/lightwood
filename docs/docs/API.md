#API

## Predictor
Lightwood has one main class; the **Predictor**, which is what you train from your data, and what you get predictions from.


```python
from lightwood import Predictor

```

### Building Blocks

The main concept is that ML models can be very modular, and each building block is something that you can thinker with. There are 3 main building blocks (*features, encoders, mixers*).

![building_blocks](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/image3-1068x927.png)


* **Features**:
    * **input_features**: These are the columns in your dataset that you want to take as input for your predictor.
    * **output_features**: These are the columns in your dataset that you want to learn how to predict.
* **Encoders**: These are tools to turn the data in your input or output features into vector/tensor representations and vice-versa.
* **Mixers**: How you mix the output of encoded features and also other mixers


## Constructor

```python

my_predictor = Predictor( output=[] | config={...} | load_from_path=<file_path>)

```
It can take on of the following three arguments:

* **load_from_path**: If you have a saved predictor that you want to load, just give the path to the file
* **output**: A list with the column names you want to predict. (*Note: If you pass this argument, lightwood will simply try to guess the best config possible*)
* **config**: A dictionary, containing the configuration on how to glue all the building blocks. 

### Predictror's **config**

The config argument allows you to pass a dictionary that defines and gives you absolute control over how to build your predictive model.
A config example goes as follows:
```python
from lightwood import COLUMN_DATA_TYPES, BUILTIN_MIXERS

config = {

        'input_features': [
            {
                'name': 'sensor1',
                'type': COLUMN_DATA_TYPES.NUMERIC
            },
            {
                'name': 'sensor2',
                'type': COLUMN_DATA_TYPES.NUMERIC
            }
        ],

        'output_features': [
            {
                'name': 'action_to_take',
                'type': COLUMN_DATA_TYPES.CATEGORICAL
            }
        ],
        
        'default_mixer': {
            'class': BUILTIN_MIXERS.NnMixer
        }
        
    }
```






#### Features

Both **input_features** and **output_features** configs are simple dicts that have the following schema

```python
{
    'name': str,
    Optional('type'): any of COLUMN_DATA_TYPES,
    Optional('encoder_class'): object,
    Optional('encoder_attrs'): dict
}
```

* **name**: is the name of the column as it is in the input data frame
* **type**: is the type od data contained. Where out of the box, supported COLUMN_DATA_TYPES are:

```python
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    DATETIME = 'datetime'
    IMAGE = 'image'
    TEXT = 'text'
    TIME_SERIES = 'time_series'
```

If you specify the type, it will use the default encoder for that type or else you can specify the encoder you want to use. 

* **encoder_class**: This is if you want to replace the default encoder with a different one, so you put the encoder class there
* **encoder_attrs**: These are the attributes that you want to setup on the encoder once the class its initialized 


#### Mixers