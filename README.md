# Lightwood

<!--- badges here? --->

Lightwood is an AutoML framework that enables you to generate and customize machine learning pipelines declarative syntax called JSON-AI.

Our goal is to make the data science/machine learning (DS/ML) life cycle easier by allowing users to focus on **what** they want to do their data without needing to write repetitive boilerplate code around machine learning and data preparation. Instead, we enable you to focus on the parts of a model that are truly unique and custom.

Lightwood works with a variety of data types such as numbers, dates, categories, tags, text, arrays and various multimedia formats. These data types can be combined together to solve complex problems. We also support a time-series mode for problems that have between-row dependencies.

Our JSON-AI syntax allows users to change any and all parts of the models Lightwood automatically generates. The syntax outlines the specifics details in each step of the modeling pipeline. Users may override default values (for example, changing the type of a column) or alternatively, entirely replace steps with their own methods (ex: use a random forest model for a predictor). Lightwood creates a "JSON-AI" object from this syntax which can then be used to automatically generate python code to represent your pipeline.

For details on how to generate JSON-AI syntax and how Lightwood works, check out the [Lightwood Philosophy](#Lightwood-Philosophy).

## Lightwood Philosophy

Lightwood abstracts the ML pipeline into 3 core steps:

(1) Pre-processing and data cleaning <br>
(2) Feature engineering <br>
(3) Model building and training <br>

<p align="center">
<img src="/assets/lightwood.png" alt="Lightwood internals" width="800"/>
</p>

#### i) Pre-processing and cleaning
For each column in your dataset, Lightwood will identify the suspected data type (numeric, categorical, etc.) via a brief statistical analysis. From this, it will generate a JSON-AI syntax. 

If the user keeps default behavior, Lightwood will perform a brief pre-processing approach to clean each column according to its identified data type. From there, it will split the data into train/dev/test splits.

The `cleaner` and `splitter` objects respectively refer to the pre-processing and the data splitting functions.

#### ii) Feature Engineering
Data can be converted into features via "encoders". Encoders represent the rules for transforming pre-processed data into a numerical representations that a model can be used. 

Encoders can be **rule-based** or **learned**. A rule-based encoder transforms data per a specific set of instructions (ex: normalized numerical data) whereas a learned encoder produces a representation of the data after training (ex: a "\[CLS\]" token in a language model).

Encoders are assigned to each column of data based on the data type; users can override this assignment either at the column-based level or at the data-type based level. Encoders inherit from the `BaseEncoder` class. 

#### iii) Model Building and Training
We call a predictive model that intakes *encoded* feature data and outputs a prediction for the target of interest a `mixer` model. Users can either use Lightwood's default mixers or create their own approaches inherited from the `BaseMixer` class.

We predominantly use PyTorch based approaches, but can support other models.

## Usage

We invite you to check out our [documentation](https://mindsdb.github.io/lightwood/) for specific guidelines and tutorials! Please stay tuned for updates and changes. 

### Quick use cases
Lightwood works with `pandas.DataFrames`. Once a DataFrame is loaded, defined a "ProblemDefinition" via a dictionary. The only thing a user needs to specify is the name of the column to predict (via the key `target`).

Create a JSON-AI syntax from the command `json_ai_from_problem`. Lightwood can then use this object to *automatically generate python code filling in the steps of the ML pipeline* via `code_from_json_ai`. 

You can make a `Predictor` object, instantiated with that code via `predictor_from_code`. 

To train a `Predictor` end-to-end, starting with unprocessed data, users can use the `predictor.learn()` command with the data.

```python
import pandas as pd
from lightwood.api.high_level import (
    ProblemDefinition,
    json_ai_from_problem,
    code_from_json_ai,
    predictor_from_code,
)

if __name__ == '__main__':
    # Load a pandas dataset
    df = pd.read_csv("https://raw.githubusercontent.com/mindsdb/benchmarks/main/benchmarks/datasets/hdi/data.csv"
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
    # print(json_ai.to_json())

    # Generate python code
    code = code_from_json_ai(json_ai)

    # OPTIONAL - see generated code
    # print(code)

    # Create a predictor from python code
    predictor = predictor_from_code(code)

    # Train a model end-to-end from raw data to a finalized predictor
    predictor.learn(df)

    # Make the train/test splits and show predictions for a few examples
    test_df = predictor.split(predictor.preprocess(df))["test"]
    preds = predictor.predict(test_df).iloc[:10]
    print(preds)
```

### BYOM: Bring your own models

Lightwood supports user architectures/approaches so long as you follow the abstractions provided within each step. 

Our [tutorials](https://mindsdb.github.io/lightwood/tutorials.html) provide specific use cases for how to introduce customization into your pipeline. Check out "custom cleaner", "custom splitter", "custom explainer", and "custom mixer". Stay tuned for further updates.


## Installation

You can install Lightwood as follows:

```python
pip3 install lightwood
```
>Note: depending on your environment, you might have to use pip instead of pip3 in the above command.

However, we recommend creating a python virtual environment.

#### Setting up a dev environment
- Python version should be in the range >=3.8, < 3.11
- Clone lightwood
- `cd lightwood && pip install -r requirements.txt && pip install -r requirements_image.txt`
- Add it to your python path (e.g. by adding `export PYTHONPATH='/where/you/cloned/lightwood':$PYTHONPATH` as a newline at the end of your `~/.bashrc` file)
- Check that the `unittest`s are passing by going into the directory where you cloned lightwood and running: `python -m unittest discover tests` 

> If `python` default to python2.x on your environment use `python3` and `pip3` instead

Currently, the preferred environment for working with lightwood is visual studio code, a very popular python IDE. However, any IDE should work. While we don't have guides for those, please feel free to use the following section as a template for VSCode, or to contribute your own tips and tricks to set up other IDEs.

#### Setting up a VSCode environment

* Install and enable setting sync using github account (if you use multiple machines)
* Install pylance (for types) and make sure to disable pyright
* Go to `Python > Lint: Enabled` and disable everything *but* flake8
* Set `python.linting.flake8Path` to the full path to flake8 (which flake8)
* Set `Python › Formatting: Provider` to autopep8
* Add `--global-config=<path_to>/lightwood/.flake8` and `--experimental` to `Python › Formatting: Autopep8 Args`
* Install live share and live share whiteboard


<!--- CONTRIBUTING.md ---->

## Contribute to Lightwood

We love to receive contributions from the community and hear your opinions! We want to make contributing to Lightwood as easy as it can be.

Being part of the core Lightwood team is possible to anyone who is motivated and wants to be part of that journey!

Please continue reading this guide if you are interested in helping democratize machine learning.

### How can you help us?

* Report a bug
* Improve documentation
* Solve an issue
* Propose new features
* Discuss feature implementations
* Submit a bug fix
* Test Lightwood with your own data and let us know how it went!

### Code contributions
In general, we follow the ["fork-and-pull"](https://docs.github.com/en/github/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model) git workflow. Here are the steps:

1. Fork the Lightwood repository
2. Make changes and commit them 
3. Make sure that the CI tests pass. You can run the test suite locally with `flake8 .` to check style and `python -m unittest discover tests` to run the automated tests. This doesn't guarantee it will pass remotely since we run on multiple envs, but should work in most cases.
4. Push your local branch to your fork
5. Submit a pull request from your repo to the `main` branch of `mindsdb/lightwood` so that we can review your changes. Be sure to merge the latest from main before making a pull request!

> Note: You will need to sign a CLI agreement for the code since lightwood is under a GPL license. 

### Feature and Bug reports
We use GitHub issues to track bugs and features. Report them by opening a [new issue](https://github.com/mindsdb/lightwood/issues/new/choose) and fill out all of the required inputs.

### Code review process
Pull request (PR) reviews are done on a regular basis. **If your PR does not address a previous issue, please make an issue first**.

If your change has a chance to affecting performance we will run our private benchmark suite to validate it.

Please, make sure you respond to our feedback/questions.

# Community
If you have additional questions or you want to chat with MindsDB core team, you can join our community: <a href="https://join.slack.com/t/mindsdbcommunity/shared_invite/zt-o8mrmx3l-5ai~5H66s6wlxFfBMVI6wQ" target="_blank"><img src="https://img.shields.io/badge/slack-@mindsdbcommunity-blueviolet.svg?logo=slack " alt="MindsDB Community"></a>.

To get updates on Lightwood and MindsDB’s latest announcements, releases, and events, sign up for our [Monthly Community Newsletter](https://mindsdb.com/newsletter/?utm_medium=community&utm_source=github&utm_campaign=lightwood%20repo).

Join our mission of democratizing machine learning and allowing developers to become data scientists!

## Contributor Code of Conduct
Please note that this project is released with a [Contributor Code of Conduct](https://github.com/mindsdb/lightwood/blob/main/CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.


# Current contributors 

<a href="https://github.com/mindsdb/lightwood/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=mindsdb/lightwood" />
</a>

# License ![PyPI - License](https://img.shields.io/pypi/l/lightwood)

* [Lightwood License](https://github.com/mindsdb/lightwood/blob/master/LICENSE)

