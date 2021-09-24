from typing import List, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from lightwood.helpers.general import evaluate_accuracy


class BestOf(BaseEnsemble):
    best_index: int

    def __init__(self, target, mixers: List[BaseMixer], data: List[EncodedDs], accuracy_functions,
                 ts_analysis: Optional[dict] = None) -> None:
        super().__init__(target, mixers, data)
        # @TODO: Need some shared accuracy functionality to determine mixer selection here
        self.maximize = True
        best_score = -pow(2, 32) if self.maximize else pow(2, 32)
        ds = ConcatedEncodedDs(data)
        for idx, mixer in enumerate(mixers):
            score_dict = evaluate_accuracy(
                ds.data_frame,
                mixer(ds)['prediction'],
                target,
                accuracy_functions,
                ts_analysis=ts_analysis
            )
            avg_score = np.mean(list(score_dict.values()))
            log.info(f'Mixer {type(mixer).__name__} obtained a best-of evaluation score of {round(avg_score,4)}')
            if self.improves(avg_score, best_score, accuracy_functions):
                best_score = avg_score
                self.best_index = idx

        self.supports_proba = self.mixers[self.best_index].supports_proba
        log.info(f'Picked best mixer: {type(self.mixers[self.best_index]).__name__}')

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        return self.mixers[self.best_index](ds, predict_proba=predict_proba)

    def improves(self, new, old, functions):
        return new > old if self.maximize else new < old
