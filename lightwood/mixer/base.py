import pandas as pd

from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments


class BaseMixer:
    """
    Base class for all mixers.

    Mixers are the backbone of all Lightwood machine learning models. They intake encoded feature representations for every column, and are tasked with learning to fulfill the predictive requirements stated in a problem definition.
    
    There are two important methods for any mixer to work:
        1. `fit()` contains all logic to train the mixer with the training data that has been encoded by all the (already trained) Lightwood encoders for any given task.
        2. `__call__()` is executed to generate predictions once the mixer has been trained using `fit()`. 
    
    An additional `partial_fit()` method is used to update any mixer that has already been trained.

    Class Attributes:
    - stable: If set to `True`, this mixer should always work. Any mixer with `stable=False` can be expected to fail under some circumstances.
    - fit_data_len: Length of the training data.
    - supports_proba: For classification tasks, whether the mixer supports yielding per-class scores rather than only returning the predicted label. 

    """  # noqa
    stable: bool
    fit_data_len: int  # @TODO (Patricio): should this really be in `BaseMixer`?
    supports_proba: bool

    def __init__(self, stop_after: float):
        """
        :param stop_after: Time budget to train this mixer.
        """
        self.stop_after = stop_after
        self.supports_proba = False

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits/trains a mixer with training data. 
         
        :param train_data: encoded representations of the training data subset. 
        :param dev_data: encoded representations of the "dev" data subset. This can be used as an internal validation subset (e.g. it is used for early stopping in the default `Neural` mixer). 
         
        """  # noqa
        raise NotImplementedError()

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Calls a trained mixer to predict the target column given some input data.
        
        :param ds: encoded representations of input data.
        :param args: a `lightwood.api.types.PredictionArguments` object, including all relevant inference-time arguments to customize the behavior.
        :return: 
        """  # noqa
        raise NotImplementedError()

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Partially fits/trains a mixer with new training data. This is a somewhat experimental method, and it aims at updating pre-existing Lightwood predictors. 

        :param train_data: encoded representations of the new training data subset. 
        :param dev_data: encoded representations of new the "dev" data subset. As in `fit()`, this can be used as an internal validation subset. 

        """  # noqa
        pass
