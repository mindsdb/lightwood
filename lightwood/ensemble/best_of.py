from typing import List, Optional

import numpy as np
import pandas as pd
from mindsdb_evaluator import evaluate_accuracies

from lightwood.helpers.log import log
from type_infer.helpers import is_nan_numeric
from lightwood.mixer.base import BaseMixer
from lightwood.ensemble.base import BaseEnsemble
from lightwood.api.types import PredictionArguments, SubmodelData
from lightwood.data.encoded_ds import EncodedDs

# special dispatches
from lightwood.mixer import GluonTSMixer  # imported from base mixer folder as it needs optional dependencies


class BestOf(BaseEnsemble):
    """
    This ensemble acts as a mixer selector. 
    After evaluating accuracy for all internal mixers with the validation data, it sets the best mixer as the underlying model.
    """  # noqa
    indexes_by_accuracy: List[float]

    def __init__(self, target, mixers: List[BaseMixer], data: EncodedDs, accuracy_functions,
                 args: PredictionArguments, ts_analysis: Optional[dict] = None, fit: bool = True) -> None:
        super().__init__(target, mixers, data, fit=False)
        self.special_dispatch_list = [GluonTSMixer]

        if fit:
            score_list = []
            for _, mixer in enumerate(self.mixers):

                if type(mixer) in self.special_dispatch_list:
                    avg_score = self.special_dispatch(mixer, data, args, target, ts_analysis)
                else:
                    score_dict = evaluate_accuracies(
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

                self.submodel_data.append(SubmodelData(
                    name=type(mixer).__name__,
                    accuracy=avg_score,
                    is_best=False
                ))

                score_list.append(avg_score)

            self.indexes_by_accuracy = list(reversed(np.array(score_list).argsort()))
            self.supports_proba = self.mixers[self.indexes_by_accuracy[0]].supports_proba
            log.info(f'Picked best mixer: {type(self.mixers[self.indexes_by_accuracy[0]]).__name__}')
            self.prepared = True
            self.submodel_data[self.indexes_by_accuracy[0]].is_best = True

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        assert self.prepared
        if args.all_mixers:
            predictions = {}
            for mixer in self.mixers:
                predictions[f'__mdb_mixer_{type(mixer).__name__}'] = mixer(ds, args=args)['prediction']
            return pd.DataFrame(predictions)
        else:
            for mixer_index in self.indexes_by_accuracy:
                mixer = self.mixers[mixer_index]
                try:
                    return mixer(ds, args=args)
                except Exception as e:
                    if mixer.stable:
                        raise(e)
                    else:
                        log.warning(f'Unstable mixer {type(mixer).__name__} failed with exception: {e}.\
                        Trying next best')

    def special_dispatch(self, mixer, data, args, target, ts_analysis):
        if isinstance(mixer, GluonTSMixer):
            return mixer.model_train_stats.loss_history[-1]
