from typing import List, Optional

import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from type_infer.dtype import dtype


class MeanEnsemble(BaseEnsemble):
    """
    When called, this ensemble will return the mean prediction from the entire list of underlying mixers.

    NOTE: can only be used in regression tasks.
    """
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict,
                 fit: Optional[bool] = True, **kwargs) -> None:
        super().__init__(target, mixers, data, fit=False)
        if dtype_dict[target] not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(
                f'This ensemble can only be used regression problems! Got target dtype {dtype_dict[target]} instead!')

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        predictions_df = pd.DataFrame()
        for mixer in self.mixers:
            predictions_df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']

        return pd.DataFrame(predictions_df.mean(axis='columns'), columns=['prediction'])

