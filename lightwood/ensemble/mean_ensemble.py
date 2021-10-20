from typing import List

import pandas as pd

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood import dtype


class MeanEnsemble(BaseEnsemble):
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict) -> None:
        super().__init__(target, mixers, data)
        if dtype_dict[target] not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(
                f'This ensemble can only be used regression problems! Got target dtype {dtype_dict[target]} instead!')

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        predictions_df = pd.DataFrame()
        for mixer in self.mixers:
            predictions_df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']

        return pd.DataFrame(predictions_df.mean(axis='columns'), columns=['prediction'])

