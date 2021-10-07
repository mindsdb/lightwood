import dill
from typing import Dict

import pandas as pd

from lightwood.api.types import ModelAnalysis


# Interface that must be respected by predictor objects generated from JSON ML and/or compatible with Mindsdb
class PredictorInterface:
    """
    Abstraction of a Lightwood predictor. The PredictorInterface encompasses how Lightwood interacts with the full ML pipeline. Internally,

    The ``PredictorInterface`` class must have 5 expected functions:

    - ``learn``: An end-to-end technique specifying how to pre-process, featurize, and train the model(s) of interest. The expected input is raw, untrained data. No explicit output is provided, but the Predictor object will "host" the trained model thus.
    - ``adjust``: The manner to incorporate new data to update pre-existing model(s).
    - ``predict``: Deploys the chosen best model, and evaluates the given data to provide target estimates.
    - ``save``: Saves the Predictor object for further use.

    The ``PredictorInterface`` is created via J{ai}son's custom code creation. A problem inherits from this class with pre-populated routines to fill out expected results, given the nature of each problem type.
    """ # noqa

    model_analysis: ModelAnalysis = None

    def __init__(self):
        pass

    def learn(self, data: pd.DataFrame) -> None:
        """
        Trains the attribute model starting from raw data. Raw data is pre-processed and cleaned accordingly. As data is assigned a particular type (ex: numerical, categorical, etc.), the respective feature encoder will convert it into a representation useable for training ML models. Of all ML models requested, these models are compiled and fit on the training data.

        :param data: Data used in training the model(s).

        :returns: Provides best fit model.
        """ # noqa
        pass

    def adjust(self, data: pd.DataFrame) -> None:
        """
        Adjusts a previously trained model on new data. Adopts the same process as ``learn`` but with the exception that the `adjust` function expects the best model to have been already trained.

        ..warnings:: Not tested yet - this is an experimental feature
        :param data: New data used to adjust a previously trained model.

        :returns: Adjusts best-fit model
        """ # noqa
        pass

    def predict(self, data: pd.DataFrame, args: Dict[str, object] = {}) -> pd.DataFrame:
        """
        Intakes raw data to provide predicted values for your trained model.

        :param data: Data (n_samples, n_columns) that the model(s) will evaluate on and provide the target prediction.
        :param args: parameters needed to update the predictor ``PredictionArguments`` object, which holds any parameters relevant for prediction.

        :returns: A dataframe of predictions of the same length of input.
        """  # noqa
        pass

    def save(self, file_path: str) -> None:
        """
        With a provided file path, saves the Predictor instance for later use.

        :param file_path: Location to store your Predictor Instance.

        :returns: Saves Predictor instance.
        """
        with open(file_path, "wb") as fp:
            dill.dump(self, fp)
