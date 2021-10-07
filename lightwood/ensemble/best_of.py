from typing import List, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.helpers.numeric import can_be_nan_numeric
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.general import evaluate_accuracy


class BestOf(BaseEnsemble):
    indexes_by_accuracy: List[float]

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, accuracy_functions,
                 ts_analysis: Optional[dict] = None) -> None:
        super().__init__(target, mixers, data)

        score_list = []
        for _, mixer in enumerate(mixers):
            score_dict = evaluate_accuracy(
                data.data_frame,
                mixer(data)['prediction'],
                target,
                accuracy_functions,
                ts_analysis=ts_analysis
            )
            avg_score = np.mean(list(score_dict.values()))
            log.info(f'Mixer: {type(mixer).__name__} got accuracy: {avg_score}')

            if can_be_nan_numeric(avg_score):
                avg_score = -pow(2, 63)
                log.warning(f'Change the accuracy of mixer {type(mixer).__name__} to valid value: {avg_score}')

            score_list.append(avg_score)

        self.indexes_by_accuracy = list(reversed(np.array(score_list).argsort()))
        self.supports_proba = self.mixers[self.indexes_by_accuracy[0]].supports_proba
        log.info(f'Picked best mixer: {type(self.mixers[self.indexes_by_accuracy[0]]).__name__}')

    def __call__(self, ds: EncodedDs, predict_proba: bool = False) -> pd.DataFrame:
        for mixer_index in self.indexes_by_accuracy:
            try:
                return self.mixers[mixer_index](ds, predict_proba=predict_proba)
            except Exception as e:
                if self.mixers[mixer_index].stable:
                    raise(e)
                else:
                    log.warning(f'Unstable mixer {type(self.mixers[mixer_index]).__name__} failed with exception: {e}.\
                    Trying next best')
