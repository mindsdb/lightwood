# Lightwood

## Lightwood is a Pytorch based framework with two objectives:

- Make it so simple that you can build predictive models with a line of code.
- Make it so flexible that you can change and customize everything.

Lightwood was inspired on Keras+Ludwig but runs on Pytorch and gives you full control of what you can do.

## Prerequisites
Python >=3.6 64bit version

## Installing Lightwood
You can install Lightwood using pip:

```
pip3 install lightwood
```

If this fails, please report the bug on github and try installing the current master branch:

```
git clone git@github.com:mindsdb/lightwood.git;
cd lightwood;
pip install --no-cache-dir -e .
```
####  Please note that, depending on your os and python setup, you might want to use pip instead of pip3.

You need python 3.6 or higher.

Note on MacOS, you need to install libomp:

```
brew install libomp
```
## Install using virtual environment
We suggest you to install Lightwood on a virtual environment to avoid dependency issues. Make sure your Python version is >=3.6. To set up a virtual environment:

## Install on Windows
Install the latest version of pip:

```
python -m pip install --upgrade pip
pip --version
```
Activate your virtual environment and install lightwood:

```
py -m pip install --user virtualenv
.\env\Scripts\activate
pip install lightwood
```
You can also use python instead of py

Install on Linux or macOS

Before installing Lightwood in a virtual environment you need to first create and activate the venv:

``
python -m venv env
source env/bin/activate
pip install lightwood
``

## Learn
You can train a Predictor as follows:

```
from lightwood import Predictor
import pandas

sensor3_predictor = Predictor(output=['sensor3'])
sensor3_predictor.learn(from_data=pandas.read_csv('sensor_data.csv'))
```

## Predict
You can now be given new readings from sensor1 and sensor2 predict what sensor3 will be.

```
prediction = sensor3_predictor.predict(when={'sensor1':1, 'sensor2':-1})
print(prediction)
```
Of course, that example was just the tip of the iceberg, please read below about the main concepts of lightwood, the API and then jump into examples.

## Lightwood Predictor API
```
from lightwood import Predictor
```
Lightwood has one main class; The Predictor, which is a modular construct that you can train and get predictions from. It is made out of 3 main building blocks (features, encoders, mixers) that you can configure, modify and expand as you wish.

![lightwood-ludwig](https://user-images.githubusercontent.com/86911142/136836861-89319684-7439-40fa-88b7-4549cb1647fa.png)



## Constructor, __init__()
```
my_predictor = Predictor( output=[] | config={...} | load_from_path=<file_path>)
```

## config
The config argument allows you to pass a dictionary that defines and gives you absolute control over how to build your predictive model. A config example goes as follows:

```
from lightwood import COLUMN_DATA_TYPES, BUILTIN_MIXERS, BUILTIN_ENCODERS

config = {

        ## REQUIRED:
        'input_features': [

            # by default each feature has an encoder, so all you have to do is specify the data type
            {
                'name': 'sensor1',
                'type': COLUMN_DATA_TYPES.NUMERIC
            },
            {
                'name': 'sensor2',
                'type': COLUMN_DATA_TYPES.NUMERIC
            },

            # some encoders have attributes that can be specified on the configuration
            # in this particular lets assume we have a photo of the product, we would like to encode this image and optimize for speed
            {
                'name': 'product_photo',
                'type': COLUMN_DATA_TYPES.IMAGE,
                'encoder_class': BUILTIN_ENCODERS.Image.Img2VecEncoder, # note that this is just a class, you can build your own if you wish
                'encoder_attrs': {
                    'aim': 'speed' 
                    # you can check the encoder attributes here: 
                    #  https://github.com/mindsdb/lightwood/blob/master/lightwood/encoders/image/img_2_vec.py
                }
            }
        ],

        'output_features': [
            {
                'name': 'action_to_take',
                'type': COLUMN_DATA_TYPES.CATEGORICAL
            }
        ],

        ## OPTIONAL
        'mixer': {
            'class': BUILTIN_MIXERS.NnMixer
        }

    }
    
    
   ```
### features
Both input_features and output_features configs are simple dicts that have the following schema

```
{
    'name': str,
    Optional('type'): any of COLUMN_DATA_TYPES,
    Optional('encoder_class'): object,
    Optional('encoder_attrs'): dict
}
```

### mixer
The default_mixer key, provides information as to what mixer to use. The schema for this variable is as follows:

```
mixer_schema = Schema({
    'class': object,
    Optional('attrs'): dict
})
```
- class: It's the actual class, that defines the Mixer, you can use any of the BUILTIN_MIXERS or pass your own.
- attrs: This is a dictionary containing the attributes you want to replace on the mixer object once its initialized. We do this, so you have maximum flexibility as to what you can customize on your Mixers.
### learn()
```
my_predictor.learn(from_data=pandas_dataframe)
```
This method is used to make the predictor learn from some data, thus the learn method takes the following arguments.

### predict()

```
my_predictor.predict(when={..} | when_data=pandas_dataframe)
```
This method is used to make predictions and it can take one of the following arguments

### save()

```
my_predictor.save(path_to=string to path)
```
Use this method to save the predictor into a desired path

### calculate_accuracy()
```
print(my_predictor.calculate_accuracy(from_data=data_source))
```


Returns the predictors overall accuracy.
