from typing import List
import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs


class IdentityEnsemble(BaseEnsemble):
    """
    This ensemble performs no aggregation. User can define an "active mixer" and calling the ensemble will call said mixer. 
    
    Ideal for use cases with single mixers where (potentially expensive) evaluation runs are done internally, as in `BestOf`.
    """  # noqa

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, args: PredictionArguments) -> None:
        super().__init__(target, mixers, data=data)
        self._active_mixer = 0
        single_row_ds = EncodedDs(data.encoders, data.data_frame.iloc[[0]], data.target)
        _ = self.mixers[self._active_mixer](single_row_ds, args)['prediction']  # prime mixer for storage, needed because NHitsMixer.model (neuralforecast.NHITS) is not serializable without this, oddly enough. Eventually, check this again and remove if possible!  # noqa
        self.prepared = True

    def __call__(self, ds: EncodedDs, args: PredictionArguments = None) -> pd.DataFrame:
        assert self.prepared
        mixer = self.mixers[self.active_mixer]
        return mixer(ds, args=args)

    @property
    def active_mixer(self):
        return self._active_mixer

    @active_mixer.setter
    def active_mixer(self, idx):
        assert 0 <= idx < len(self.mixers), f'The ensemble has {len(self.mixers)} mixers, please provide a valid index.'
        self._active_mixer = idx
