from copy import deepcopy
from types import SimpleNamespace
from typing import Dict

import numpy as np
from sklearn.utils import shuffle

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.api.types import PredictionArguments


class PermutationFeatureImportance(BaseAnalysisBlock):
    """
    Analysis block that estimates column importances via permutation.

    Roughly speaking, the procedure:
        - iterates over all input columns
        - if the input column is optional, shuffle its values, then generate predictions for the input data
        - compare this accuracy with the accuracy obtained using unshuffled data
        - all accuracy differences are normalized with respect to the original accuracy (clipped at zero if negative)
        - report these as estimated column importance scores

    Note that, crucially, this method does not refit the predictor at any point.

    Reference:
        https://compstat-lmu.github.io/iml_methods_limitations/pfi.html
        https://scikit-learn.org/stable/modules/permutation_importance.html
    """
    def __init__(self, disable_column_importance=False, deps=tuple('AccStats',)):
        super().__init__(deps=deps)
        self.disable_column_importance = disable_column_importance
        self.n_decimals = 3

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if self.disable_column_importance or ns.tss.is_timeseries or ns.has_pretrained_text_enc:
            info['column_importances'] = None
        else:
            shuffled_col_accuracy = {}
            shuffled_cols = []
            for x in ns.input_cols:
                if ('__mdb' not in x) and \
                        (not ns.tss.is_timeseries or (x != ns.tss.order_by and x not in ns.tss.historical_columns)):
                    shuffled_cols.append(x)

            for col in shuffled_cols:
                partial_data = deepcopy(ns.encoded_val_data)
                partial_data.clear_cache()
                partial_data.data_frame[col] = shuffle(partial_data.data_frame[col].values)

                args = {'predict_proba': True} if ns.is_classification else {}
                shuffled_preds = ns.predictor(partial_data, args=PredictionArguments.from_dict(args))

                shuffled_col_accuracy[col] = np.mean(list(evaluate_accuracy(
                    ns.data,
                    shuffled_preds['prediction'],
                    ns.target,
                    ns.accuracy_functions
                ).values()))

            column_importances = {}
            acc_increases = np.zeros((len(shuffled_cols),))
            for i, col in enumerate(shuffled_cols):
                accuracy_increase = (info['normal_accuracy'] - shuffled_col_accuracy[col])
                acc_increases[i] = round(accuracy_increase, self.n_decimals)
            for col, inc in zip(shuffled_cols, acc_increases):
                column_importances[col] = inc  # scores go from 0 to 1

            info['column_importances'] = column_importances

        return info
