from typing import List, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.helpers.numeric import can_be_nan_numeric
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.general import evaluate_accuracy


class WeightedMeanEnsemble(BaseEnsemble):
    weights: List[float]

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, accuracy_functions,
                 args: PredictionArguments, ts_analysis: Optional[dict] = None) -> None:
        super().__init__(target, mixers, data)

        score_list = []
        for _, mixer in enumerate(mixers):
            score_dict = evaluate_accuracy(
                data.data_frame,
                mixer(data, args)['prediction'],
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

        self.weights = list(self.accuracies_to_weights(np.array(score_list)))
        self.supports_proba = True
        for mixer in self.mixers:
            self.supports_proba = self.supports_proba and mixer.supports_proba

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        df = pd.DataFrame()
        for mixer in self.mixers:
            df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']
        avg_predictions = np.average(df, weights=self.weights, axis=1, dtype='float')
        return pd.DataFrame(avg_predictions, columns=['prediction'])

    def accuracies_to_weights(self, x: np.array) -> np.array:
        # Converts accuracies to weights using the softmax function.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
