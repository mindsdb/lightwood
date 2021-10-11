.. -*- coding: utf-8 -*-
.. lightwood_docs documentation master file, created by
   sphinx-quickstart on Tue Sep  7 13:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

****************************************
Welcome to Lightwood's Documentation!
****************************************
:Release: |release|
:Date: |today|

.. warning::
   This project is under development; consider features in beta.
   
.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:

   lightwood_philosophy
   tutorials
   api

Lightwood is an AutoML framework that enables you to generate and customize machine learning pipelines declarative syntax called JSON-AI.

Our goal is to make the data science/machine learning (DS/ML) life cycle easier by allowing users to focus on **what** they want to do their data without needing to write repetitive boilerplate code around machine learning and data preparation. Instead, we enable you to focus on the parts of a model that are truly unique and custom.

Lightwood works with a variety of data types such as numbers, dates, categories, tags, text, arrays and various multimedia formats. These data types can be combined together to solve complex problems. We also support a time-series mode for problems that have between-row dependencies.

Our JSON-AI syntax allows users to change any and all parts of the models Lightwood automatically generates. The syntax outlines the specifics details in each step of the modeling pipeline. Users may override default values (for example, changing the type of a column) or alternatively, entirely replace steps with their own methods (ex: use a random forest model for a predictor). Lightwood creates a "JSON-AI" object from this syntax which can then be used to automatically generate python code to represent your pipeline.

For details on how to generate JSON-AI syntax and how Lightwood works, check out the #TODO LIGHTWOODPHILO[Lightwood Philosophy](#Lightwood-Philosophy).

Installation
============

You can install Lightwood as follows:

:: 
   pip3 install lightwood

.. note:: depending on your environment, you might have to use pip instead of pip3 in the above command.

However, we recommend creating a python virtual environment.

Setting up a dev environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clone lightwood
- `cd lightwood && pip install requirements.txt`
- Add it to your python path (e.g. by adding `export PYTHONPATH='/where/you/cloned/lightwood':$PYTHONPATH` as a newline at the end of your `~/.bashrc` file)
- Check that the `unittest`s are passing by going into the directory where you cloned lightwood and running: `python -m unittest discover tests` 

> If `python` default to python2.x on your environment use `python3` and `pip3` instead

Currently, the preferred environment for working with lightwood is visual studio code, a very popular python IDE. However, any IDE should work. While we don't have guides for those, please feel free to use the following section as a template for VSCode, or to contribute your own tips and tricks to set up other IDEs.

Setting up a VSCode environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Install and enable setting sync using github account (if you use multiple machines)
* Install pylance (for types) and make sure to disable pyright
* Go to `Python > Lint: Enabled` and disable everything *but* flake8
* Set `python.linting.flake8Path` to the full path to flake8 (which flake8)
* Set `Python › Formatting: Provider` to autopep8
* Add `--global-config=<path_to>/lightwood/.flake8` and `--experimental` to `Python › Formatting: Autopep8 Args`
* Install live share and live share whiteboard


Example Use Cases
=======================

Lightwood works with `pandas.DataFrames`. Once a DataFrame is loaded, defined a "ProblemDefinition" via a dictionary. The only thing a user needs to specify is the name of the column to predict (via the key `target`).

Create a JSON-AI syntax from the command `json_ai_from_problem`. Lightwood can then use this object to *automatically generate python code filling in the steps of the ML pipeline* via `code_from_json_ai`. 

You can make a `Predictor` object, instantiated with that code via `predictor_from_code`. 

To train a `Predictor` end-to-end, starting with unprocessed data, users can use the `predictor.learn()` command with the data.

:: 
   import pandas as pd
   from lightwood.api.high_level import (
       ProblemDefinition,
       json_ai_from_problem,
       code_from_json_ai,
       predictor_from_code,
   )

   # Load a pandas dataset
   df = pd.read_csv(
       "https://raw.githubusercontent.com/mindsdb/benchmarks/main/benchmarks/datasets/hdi/data.csv"
   )

   # Define the prediction task by naming the target column
   pdef = ProblemDefinition.from_dict(
       {
           "target": "Development Index",  # column you want to predict
       }
   )

   # Generate JSON-AI code to model the problem
   json_ai = json_ai_from_problem(df, problem_definition=pdef)

   # OPTIONAL - see the JSON-AI syntax
   #print(json_ai.to_json())

   # Generate python code
   code = code_from_json_ai(json_ai)

   # OPTIONAL - see generated code
   #print(code)

   # Create a predictor from python code
   predictor = predictor_from_code(code)

   # Train a model end-to-end from raw data to a finalized predictor
   predictor.learn(df)

   # Make the train/test splits and show predictions for a few examples
   test_df = predictor.split(predictor.preprocess(df))["test"]
   preds = predictor.predict(test).iloc[:10]
   print(preds)

BYOM: Bring your own models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lightwood supports user architectures/approaches so long as you follow the abstractions provided within each step. 

Our #TODO[tutorials](https://lightwood.io/tutorials.html) provide specific use cases for how to introduce customization into your pipeline. Check out "custom cleaner", "custom splitter", "custom explainer", and "custom mixer". Stay tuned for further updates.


Contribute to Lightwood
=======================

We love to receive contributions from the community and hear your opinions! We want to make contributing to Lightwood as easy as it can be.

Being part of the core Lightwood team is possible to anyone who is motivated and wants to be part of that journey!

Please continue reading this guide if you are interested in helping democratize machine learning.

How can you help us?
^^^^^^^^^^^^^^^^^^^^^^^^
* Report a bug
* Improve documentation
* Solve an issue
* Propose new features
* Discuss feature implementations
* Submit a bug fix
* Test Lightwood with your own data and let us know how it went!
