from typing import List

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.model.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from lightwood.helpers.general import evaluate_accuracy


class BestOf(BaseEnsemble):
    best_index: int

    def __init__(self, target, models: List[BaseMixer], data: List[EncodedDs], accuracy_functions) -> None:
        super().__init__(target, models, data)
        # @TODO: Need some shared accuracy functionality to determine model selection here
        self.maximize = True
        best_score = -pow(2, 32) if self.maximize else pow(2, 32)
        ds = ConcatedEncodedDs(data)
        for idx, model in enumerate(models):
            score_dict = evaluate_accuracy(
                ds.data_frame,
                model(ds)['prediction'],
                target,
                accuracy_functions
            )
            avg_score = np.mean(list(score_dict.values()))
            log.info(f'Model {type(model).__name__} obtained a best-of evaluation score of {round(avg_score,4)}')
            if self.improves(avg_score, best_score, accuracy_functions):
                best_score = avg_score
                self.best_index = idx

        self.supports_proba = self.models[self.best_index].supports_proba
        log.info(f'Picked best model: {type(self.models[self.best_index]).__name__}')

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        return self.models[self.best_index](ds, predict_proba=predict_proba)

    def improves(self, new, old, functions):
        return new > old if self.maximize else new < old
