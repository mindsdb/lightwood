MindsDB Lightwood: Machine Learning Classification Example
==========================================================

*Community Contributor:*\ `Chandre Van Der
Westhuizen <https://github.com/chandrevdw31>`__

AI/Machine learning can be intimidating, but does it *have* to be?
------------------------------------------------------------------

Presenting the easiest tutorial on machine learning classification for
beginners. This is a beginner’s guide into machine learning
classification using **Lightwood**, but can also intrigue an expert’s
interest.

Getting to know Lightwood
--------------------------

Lightwood is an AutoML framework that enables you to generate and
customize machine learning pipelines declarative syntax called JSON-AI.

MindsDB allows users to skip the redundant data preparation and
repetitive code writing so that the focus can be on what must be done
with the data. The goal is to simplify the life cycle of data
science/machine learning (DS/ML).

You will find it pleasurable that various data types are accommodated,
for example audio, videos, numbers, dates, categories, tags, text,
arrays etc. which can be combined to solve complex problems. A
time-series mode for problems that have between-row dependencies is also
supported.

Models that gets automatically generated can be changed by users via
their JSON-AI syntax,that outlines details in each step of your modeling
pipeline, from which Lightwood creates a JSON-AI object that can
automatically generate python code to represent the pipeline. Default
values can be overridden and steps can be replaced by users.

Predicting Pulsars
------------------

Pulsars are Neutron stars which emits a pattern of broadband radio waves
that spreads across the sky that can be detectable from Earth.
Scientists use these detections to measure cosmic distances and search
for other planets as well as help pin point cosmic events like
collisions between supermassive black holes by finding gravitational
waves.It can be quite tricky to detect real pulsars due to radio
frequencies and noise. Through deep learning tools, we can build a
simple classifier to predict whether a detected signal comes from pulsar
star

For this tutorial we will be predicting the presence of a pulsar. To
showcase how easy and accessible Lightwood is, I will be using an online
IDE called GoormIDE.

Pre-requisites
--------------

To follow the steps, ensure you have the following tools:

1. Access to the online IDE `GoormIDE <https://ide.goorm.io/>`__. Note
   that this is optional and any online IDE or Visual Studio can be
   used.
2. Download the dataset. This dataset can be found on
   `Kaggle <https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate?select=pulsar_data_train.csv>`__.
3. `Python 3.7+ <https://docs.python.org/3/>`__ 4.Optional- Jupyter
   Notebook.

GoormIDE
--------

GoormIDE is a cloud IDE service which makes it accessible for all users.
You can access their website `here <https://ide.goorm.io/>`__. You will
be able to sign up with an existing google or github account.

To `sign
up <https://help.goorm.io/en/goormide/01.introduction/sign-in>`__ and
`create a
container <https://help.goorm.io/en/goormide/01.introduction/dashboard>`__,
you can find and follow the guide
`here <https://help.goorm.io/en/goormide/01.introduction/create-a-new-container>`__.

Installing Lightwood
--------------------

To install Lightwood, open a terminal and run the following command:

.. code:: bash

   pip3 install lightwood

If you are experiencing any difficulties with installing
Lightwood,please use the below command:

.. code:: bash

   pip3 install lightwood --no-cache-dir

Once Lightwood is installed,open a Jupyter Notebook in Goorm. This is
optional, but highly recommended.

Executing the code
----------------

We will now insert Lightwood’s predictor code into our Jupyter Notebook.
You can access the code on Lightwood’s documentation which is available
on MindsDB’s GitHub `here <https://github.com/mindsdb/lightwood>`__

.. code:: python

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
   preds = predictor.predict(test_df).iloc[:10]
   print(preds)

We will load our pandas dataset and define the prediction task by naming
the target column.

The following parameters will be set: >For more information on
Lightwood’s parameters, click
`here <https://lightwood.io/api/types.html?highlight=problem#api.types.ProblemDefinition>`__

.. code:: python

   pdef = ProblemDefinition.from_dict(
       {
           "target": "target_class",  # column you want to predict
           "unbias_target": True
       }
   )

The code should look like this: |code|

Executing the code and the results should look similar to this:

.. figure:: https://raw.githubusercontent.com/chandrevdw31/mindsdb-tutorials/main/Assets/Pulsar/pulsar_lightwood.gif
   :alt: video

   video

Here is our example and results:

.. figure:: https://raw.githubusercontent.com/chandrevdw31/mindsdb-tutorials/main/Assets/Pulsar/pulsar_result.png
   :alt: Results

   Results

You can save your results

.. code:: python

   # Save predictor for later use

   predictor.save("./pulsars_model")

   preds.to_csv("test_predictations.csv")

And that’s it! We have covered the basics of creating and training a
predictive machine learning model in Lightwood.

You can download the Jupyter Notebook for this tutorial
`here <https://github.com/chandrevdw31/mindsdb-tutorials/blob/main/Assets/Pulsar/lightwood_pulsar_tut.ipynb>`__

How can you help us?
~~~~~~~~~~~~~~~~~~~~

-  Report a bug
-  Improve documentation
-  Solve an issue
-  Propose new features
-  Discuss feature implementations
-  Submit a bug fix
-  Test Lightwood with your own data and let us know how it went!

For more information, check out `Lightwood’s
documentation <https://lightwood.io/lightwood_philosophy.html>`__.

Want to try it out for yourself? Sign up for a `free MindsDB
account <https://cloud.mindsdb.com/signup?utm_medium=community&utm_source=ext.%20blogs&utm_campaign=blog-crop-detection>`__
and join our community! Engage with MindsDB community on
`Slack <https://join.slack.com/t/mindsdbcommunity/shared_invite/zt-o8mrmx3l-5ai~5H66s6wlxFfBMVI6wQ>`__
or `Github <https://github.com/mindsdb/mindsdb/discussions>`__ to ask
questions, share and express ideas and thoughts!

.. |code| image:: https://raw.githubusercontent.com/chandrevdw31/mindsdb-tutorials/main/Assets/Pulsar/pulsar_code.png
