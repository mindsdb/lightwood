from typing import List, Optional, Dict

import pandas as pd
import numpy as np

from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from type_infer.dtype import dtype
from lightwood.helpers.general import evaluate_accuracy
from type_infer.helpers import is_nan_numeric
from lightwood.helpers.log import log


class ModeEnsemble(BaseEnsemble):
    """
    When called, this ensemble will return the mode prediction from the entire list of underlying mixers.

    If there are multiple modes, the mode whose voting mixers have the highest score will be returned.

    NOTE: can only be used in categorical tasks.
    """
    mixer_scores: Dict[str, float]

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, dtype_dict: dict,
                 accuracy_functions, args: PredictionArguments, ts_analysis: Optional[dict] = None,
                 fit: Optional[bool] = True, **kwargs) -> None:
        super().__init__(target, mixers, data, fit=False)
        self.mixer_scores = {}

        if fit:
            if dtype_dict[target] not in (dtype.binary, dtype.categorical, dtype.tags):
                raise Exception(
                    'This ensemble can only be used in classification problems! ' +
                    f'Got target dtype {dtype_dict[target]} instead!')

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
                    avg_score = -pow(2, 63)
                    log.warning(f'Change the accuracy of mixer {type(mixer).__name__} to valid value: {avg_score}')

                self.mixer_scores[f'__mdb_mixer_{type(mixer).__name__}'] = avg_score
            self.prepared = True

    def _pick_mode_highest_score(self, prediction: pd.Series):
        """If the predictions are unimodal, return the mode. If there are multiple modes, return the mode whose voting
        mixers have the highest score."""
        prediction_counts = prediction.value_counts()

        # If there is a clear winner, i.e. only one prediction
        if len(prediction_counts) == 1:
            return prediction_counts.index[0]

        counts = prediction_counts.values  # how many times all predictions have appeared
        max_count = np.max(counts)  # how many times the most frequent predictions have apppeared

        # most frequent predictions and how many times they appeared
        modes = prediction_counts[prediction_counts == max_count]

        modes_predictions = modes.index  # most frequent predictions

        # For each mode, get the sum of the scores of the predictors who voted for it
        modes_predictions_scores = {}
        for mode_prediction in modes_predictions:
            voting_mixers_name = prediction[prediction == mode_prediction].index.tolist()
            modes_predictions_scores[mode_prediction] = np.sum(
                [self.mixer_scores[mixer_name] for mixer_name in voting_mixers_name])

        # Return the mode with the maximum sum of accuracies
        return max(modes_predictions_scores, key=modes_predictions_scores.get)

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        assert self.prepared
        predictions_df = pd.DataFrame()
        for mixer in self.mixers:
            predictions_df[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']

        mode_df = predictions_df.apply(func=self._pick_mode_highest_score, axis='columns')

        return pd.DataFrame(mode_df, columns=['prediction'])
