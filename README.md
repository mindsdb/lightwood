<h1 align="center">
	<img width="500" src="https://github.com/mindsdb/mindsdb-docs/blob/master/mindsdb-docs/docs/assets/MindsDBLightwood@3x.png" alt="Lightwood">
	<br>
	<br>
</h1>

![Lightwood Actions workflow](https://github.com/mindsdb/lightwood/workflows/Lightwood%20Actions%20workflow/badge.svg)
![](https://img.shields.io/badge/python-3.6%20|%203.7|%203.8-brightgreen.svg)
[![PyPI version](https://badge.fury.io/py/lightwood.svg)](https://badge.fury.io/py/lightwood)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightwood)


## What is lightwood?

* Lightwood is a tool for automatic machine learning
* Lightwood is an implementation of J{AI}SON
* Lightwood is a library for declarative ML

## End to End


### Generate code

Lightwood works with pandas dataframes. You can give it a dataframe and tell ask it to analyze it in order to generate code for solving some inferential problem. It will infer the "types" of your columns and run some statistical analysis on the data. Finally, once all of this is done, it will generate a `JsonAI` object.

<h1 align="center">
	<img src="https://github.com/mindsdb/lightwood/blob/staging/docs/1.jpg" alt="Lightwood">
	<br>
	<br>
</h1>

*Important note: The `JsonAI` object is just a json. You can dump it by calling `to_dict` or `to_json`, then edit it to your liking. Then you can reload it by calling `lightwood.JsonAI.from_dict` (or `from_json`). This is a rather tedious way of doing this, and we'll document J{AI}SON editing better when we have some tools to help the process along*

```
import lightwood
import requests

data = requests.get('https://raw.githubusercontent.com/mindsdb/benchmarks/main/benchmarks/datasets/hdi/data.csv').text

dataframe = pd.read_csv(io.StringIO(data), sep=",")
problem_definition = lightwood.ProblemDefinition.from_dict({'target': 'Development Index'})
```

### Train a Predictor

...


### Make some inferences

...



## Current contributors 

<a href="https://github.com/mindsdb/lightwood/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=mindsdb/lightwood" />
</a>

Made with [contributors-img](https://contributors-img.web.app).

## License ![PyPI - License](https://img.shields.io/pypi/l/lightwood)

* [Lightwood License](https://github.com/mindsdb/lightwood/blob/master/LICENSE)
