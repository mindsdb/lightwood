from typing import List, Dict, Optional

import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments


class BaseEnsemble:
    """
    Base class for all ensembles.

    Ensembles wrap sets of Lightwood mixers, with the objective of generating better predictions based on the output of each mixer. 
    
    There are two important methods for any ensemble to work:
        1. `__init__()` should prepare all mixers and internal ensemble logic.
        2. `__call__()` applies any aggregation rules to generate final predictions based on the output of each mixer.

    Notable class attributes:
    - mixers: List of mixers the ensemble will use.
    - supports_proba: For classification tasks, whether the ensemble supports yielding per-class scores rather than only returning the predicted label.

    For time series specific ensembles, a few additional steps need to be implemented:
     - Store the latest `window` rows of data used when training in `self.context`.
     - Override the `get_latest_context()` method to use the data stored in the step above.
    """  # noqa

    data: EncodedDs
    mixers: List[BaseMixer]
    dtype_dict: Dict[str, str]
    target: str
    best_index: int  # @TODO: maybe only applicable to BestOf
    supports_proba: bool
    context: pd.DataFrame

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: Dict[str, str]) -> None:
        self.data = data
        self.mixers = mixers
        self.best_index = 0
        self.supports_proba = False
        self.target = target
        self.dtype_dict = dtype_dict
        self.context = pd.DataFrame()

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        raise NotImplementedError()

    def store_context(self, data: pd.DataFrame, ts_analysis: Optional[Dict] = {}) -> None:
        """
        This method gets called during ensembling for time series tasks.
        It should store the latest `window` data points seen during training time (including any validation data) so that, by default, predictions will be made for the inmmediate next horizon without needing any input.
        """  # noqa
        pass

    def get_context(self) -> pd.DataFrame:
        """
        This method gets called during inference in time series tasks if no input is passed.
        It should retrieve data saved in self.store_context().
        """  # noqa
        return self.context
