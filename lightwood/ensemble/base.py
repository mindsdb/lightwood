from typing import List

import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import PredictionArguments, SubmodelData


class BaseEnsemble:
    """
    Base class for all ensembles.

    Ensembles wrap sets of Lightwood mixers, with the objective of generating better predictions based on the output of each mixer. 
    
    There are two important methods for any ensemble to work:
        1. `__init__()` should prepare all mixers and internal ensemble logic.
        2. `__call__()` applies any aggregation rules to generate final predictions based on the output of each mixer. 

    Class Attributes:
    - mixers: List of mixers the ensemble will use.
    - supports_proba: For classification tasks, whether the ensemble supports yielding per-class scores rather than only returning the predicted label. 

    """  # noqa
    data: EncodedDs
    mixers: List[BaseMixer]
    best_index: int  # @TODO: maybe only applicable to BestOf
    supports_proba: bool
    submodel_data: List[SubmodelData]

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs) -> None:
        self.data = data
        self.mixers = mixers
        self.best_index = 0
        self.supports_proba = False
        self.submodel_data = []

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        raise NotImplementedError()
