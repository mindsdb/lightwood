import dill
from typing import Dict, Optional

import pandas as pd
from lightwood.api.types import ModelAnalysis


# Interface that must be respected by predictor objects generated from JSON ML and/or compatible with Mindsdb
class PredictorInterface:
    """
    Abstraction of a Lightwood predictor. The ``PredictorInterface`` encompasses how Lightwood interacts with the full ML pipeline. Internally,

    The ``PredictorInterface`` class must have several expected functions:
    
    - ``analyze_data``: Peform a statistical analysis on the unprocessed data; this helps inform downstream encoders and mixers on how to treat the data types.
    - ``preprocess``: Apply cleaning functions to each of the columns within the dataset to prepare them for featurization
    - ``split``: Split the input dataset into a train/dev/test set according to your splitter function
    - ``prepare``: Create and, if necessary, train your encoders to create feature representations from each column of your data.
    - ``featurize``: For input, pre-processed data, create feature vectors
    - ``fit``: Train your mixer models to yield predictions from featurized data
    - ``analyze_ensemble``: Evaluate the quality of fit for your mixer models
    - ``adjust``: Incorporate new data to update pre-existing model(s).

    For simplification, we offer an end-to-end approach that allows you to input raw data and follow every step of the process until you reach a trained predictor with the ``learn`` function:

        - ``learn``: An end-to-end technique specifying how to pre-process, featurize, and train the model(s) of interest. The expected input is raw, untrained data. No explicit output is provided, but the Predictor object will "host" the trained model thus.
    
    You can also use the predictor to now estimate new data:

    - ``predict``: Deploys the chosen best model, and evaluates the given data to provide target estimates.
    - ``save``: Saves the Predictor object for further use.

    The ``PredictorInterface`` is created via J{ai}son's custom code creation. A problem inherits from this class with pre-populated routines to fill out expected results, given the nature of each problem type.
    """ # noqa

    model_analysis: ModelAnalysis = None

    def __init__(self):
        pass

    def analyze_data(self, data: pd.DataFrame) -> None:
        """
        Performs a statistical analysis on the data to identify distributions, imbalanced classes, and other nuances within the data.

        :param data: Data used in training the model(s).
        """ # noqa
        pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the unprocessed dataset provided.

        :param data: (Unprocessed) Data used in training the model(s).
        :returns: The cleaned data frame
        """ # noqa
        pass

    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Categorizes the data into a training/testing split; if data is a classification problem, will stratify the data.

        :param data: Pre-processed data, but generically any dataset to split into train/dev/test.
        :returns: Dictionary containing training/testing fraction
        """ # noqa
        pass

    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Prepares the encoders for each column of data.

        :param data: Pre-processed data that has been split into train/test. Explicitly uses "train" and/or "dev" in preparation of encoders.

        :returns: Nothing; prepares the encoders for learned representations.
        """  # noqa

    def featurize(self, split_data: Dict[str, pd.DataFrame]):
        """
        Provides an encoded representation for each dataset in ``split_data``. Requires `self.encoders` to be prepared.

        :param split_data: Pre-processed data from the dataset, split into train/test (or any other keys relevant)

        :returns: For each dataset provided in ``split_data``, the encoded representations of the data.
        """ # noqa
        pass

    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        """
        Fits "mixer" models to train predictors on the featurized data. Instantiates a set of trained mixers and an ensemble of them. 

        :param enc_data: Pre-processed and featurized data, split into the relevant train/test splits. Keys expected are "train", "dev", and "test"
        """  # noqa
        pass

    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        """
        Evaluate the quality of mixers within an ensemble of models.

        :param enc_data: Pre-processed and featurized data, split into the relevant train/test splits.
        """
        pass

    def learn(self, data: pd.DataFrame) -> None:
        """
        Trains the attribute model starting from raw data. Raw data is pre-processed and cleaned accordingly. As data is assigned a particular type (ex: numerical, categorical, etc.), the respective feature encoder will convert it into a representation useable for training ML models. Of all ML models requested, these models are compiled and fit on the training data.

        This step amalgates ``preprocess`` -> ``featurize`` -> ``fit`` with the necessary splitting + analyze_data that occurs. 

        :param data: (Unprocessed) Data used in training the model(s).

        :returns: Nothing; instantiates with best fit model from ensemble.
        """  # noqa
        pass

    def adjust(self, new_data: pd.DataFrame, old_data: Optional[pd.DataFrame] = None) -> None:
        """
        Adjusts a previously trained model on new data. Adopts the same process as ``learn`` but with the exception that the `adjust` function expects the best model to have been already trained.

        .. warning:: This is experimental and subject to change. 
        :param new_data: New data used to adjust a previously trained model.
        :param old_data: In some situations, the old data is still required to train a model (i.e. Regression mixer) to ensure the new data doesn't entirely override it.

        :returns: Nothing; adjusts best-fit model
        """  # noqa
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
