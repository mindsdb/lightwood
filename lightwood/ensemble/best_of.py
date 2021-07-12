from typing import List
from lightwood.model.base import BaseModel
import pandas as pd
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from lightwood.ensemble.base import BaseEnsemble
import numpy as np


class BestOf(BaseEnsemble):
    best_index: int

    def __init__(self, target, models: List[BaseModel], data: List[EncodedDs], accuracy_functions) -> None:
        super().__init__(models, data)
        # @TODO: Need some shared accuracy functionality to determine model selection here
        best_score = -pow(2, 32)
        for idx, model in enumerate(models):
            score_dict = evaluate_accuracy(
                data[target],
                model(ConcatedEncodedDs(data))['prediction'],
                accuracy_functions
            )
            avg_score = np.mean(list(score_dict.values()))
            if avg_score > best_score:
                best_score = avg_score
                self.best_index = idx

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        return self.models[self.best_index](ds)
