from copy import deepcopy
from types import SimpleNamespace
from typing import Dict

import numpy as np

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.api.types import PredictionArguments


class GlobalFeatureImportance(BaseAnalysisBlock):
    """
    Analysis block that estimates column importance with a variant of the LOCO (leave-one-covariate-out) algorithm.

    Roughly speaking, the procedure:
        - iterates over all input columns
        - if the input column is optional, then make a predict with its values set to None
        - compare this accuracy with the accuracy obtained using all data
        - all accuracy differences are normalized with respect to the original accuracy (clipped at zero if negative)
        - report these as estimated column importance scores

    Note that, crucially, this method does not refit the predictor at any point.

    Reference:
        https://compstat-lmu.github.io/iml_methods_limitations/pfi.html
    """
    def __init__(self, disable_column_importance=False, deps=tuple('AccStats',)):
        super().__init__(deps=deps)
        self.disable_column_importance = disable_column_importance

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if self.disable_column_importance or ns.tss.is_timeseries or ns.has_pretrained_text_enc:
            info['column_importances'] = None
        else:
            empty_input_accuracy = {}
            ignorable_input_cols = []
            for x in ns.input_cols:
                if ('__mdb' not in x) and \
                        (not ns.tss.is_timeseries or (x != ns.tss.order_by and x not in ns.tss.historical_columns)):
                    ignorable_input_cols.append(x)

            for col in ignorable_input_cols:
                partial_data = deepcopy(ns.encoded_val_data)
                partial_data.clear_cache()
                partial_data.data_frame[col] = [None] * len(partial_data.data_frame[col])

                args = {'predict_proba': True} if ns.is_classification else {}
                empty_input_preds = ns.predictor(partial_data, args=PredictionArguments.from_dict(args))

                empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
                    ns.data,
                    empty_input_preds['prediction'],
                    ns.target,
                    ns.accuracy_functions
                ).values()))

            column_importances = {}
            acc_increases = np.zeros((len(ignorable_input_cols),))
            for i, col in enumerate(ignorable_input_cols):
                accuracy_increase = (info['normal_accuracy'] - empty_input_accuracy[col]) / info['normal_accuracy']
                acc_increases[i] = max(0, accuracy_increase)
            acc_increases = (acc_increases / max(acc_increases))
            for col, inc in zip(ignorable_input_cols, acc_increases):
                column_importances[col] = inc  # scores go from 0 to 1

            info['column_importances'] = column_importances

        return info
