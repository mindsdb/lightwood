from typing import List, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from lightwood.helpers.numeric import is_nan_numeric
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.general import evaluate_accuracy
from lightwood import dtype


class WeightedMeanEnsemble(BaseEnsemble):
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, args: PredictionArguments,
                 dtype_dict: dict, accuracy_functions, ts_analysis: Optional[dict] = None) -> None:
        super().__init__(target, mixers, data)
        if dtype_dict[target] not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(
                f'This ensemble can only be used regression problems! Got target dtype {dtype_dict[target]} instead!')

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

            if is_nan_numeric(avg_score):
                log.warning(f'Could not compute a valid accuracy for mixer: {type(mixer).__name__}, \
                              functions: {accuracy_functions}, yielded invalid average score {avg_score}, \
                              resetting that to -pow(2,63) instead.')
                avg_score = -pow(2, 63)

            score_list.append(avg_score)

        self.weights = self.accuracies_to_weights(np.array(score_list))

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        df = pd.DataFrame()
        for mixer in self.mixers:
            df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']

        avg_predictions_df = df.apply(lambda x: np.average(x, weights=self.weights), axis='columns')
        return pd.DataFrame(avg_predictions_df, columns=['prediction'])

    def accuracies_to_weights(self, x: np.array) -> np.array:
        # Converts accuracies to weights using the softmax function.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
