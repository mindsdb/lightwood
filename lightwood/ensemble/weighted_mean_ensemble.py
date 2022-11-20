from typing import List, Optional

import numpy as np
import pandas as pd

from lightwood.helpers.log import log
from type_infer.helpers import is_nan_numeric
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.general import evaluate_accuracy
from type_infer.dtype import dtype


class WeightedMeanEnsemble(BaseEnsemble):
    """
    This ensemble determines a weight vector to return a weighted mean of the underlying mixers.

    More specifically, each model is evaluated on the validation dataset and assigned an accuracy score (as per the fixed accuracy function at the JsonAI level).

    Afterwards, all the scores are softmaxed to obtain the final weights.

    Note: this ensemble only supports regression tasks.
    """  # noqa
    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, args: PredictionArguments,
                 dtype_dict: dict, accuracy_functions, ts_analysis: Optional[dict] = None,
                 fit: Optional[bool] = True, **kwargs) -> None:
        super().__init__(target, mixers, data)
        if dtype_dict[target] not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(
                f'This ensemble can only be used regression problems! Got target dtype {dtype_dict[target]} instead!')

        if fit:
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
            self.prepared = True

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        assert self.prepared
        df = pd.DataFrame()
        for mixer in self.mixers:
            df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']

        avg_predictions_df = df.apply(lambda x: np.average(x, weights=self.weights), axis='columns')
        return pd.DataFrame(avg_predictions_df, columns=['prediction'])

    @staticmethod
    def accuracies_to_weights(x: np.array) -> np.array:
        # Converts accuracies to weights using the softmax function.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
