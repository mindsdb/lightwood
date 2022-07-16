from copy import deepcopy
from types import SimpleNamespace
from typing import Dict

import torch
import numpy as np

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.analysis.nc.util import t_softmax
from lightwood.api.types import PredictionArguments


class GlobalFeatureImportance(BaseAnalysisBlock):
    """
    Analysis block that estimates column importance with a variant of the LOCO (leave-one-covariate-out) algorithm.

    Roughly speaking, the procedure:
        - iterates over all input columns
        - if the input column is optional, then make a predict with its values set to None
        - compare this accuracy with the accuracy obtained using all data
        - all accuracy differences are passed through a softmax and reported as estimated column importance scores

    Note that, crucially, this method does not refit the predictor at any point.

    Reference:
        https://compstat-lmu.github.io/iml_methods_limitations/pfi.html
    """
    def __init__(self, disable_column_importance):
        super().__init__()
        self.disable_column_importance = disable_column_importance

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if self.disable_column_importance or ns.tss.is_timeseries or ns.has_pretrained_text_enc:
            info['column_importances'] = None
        else:
            empty_input_accuracy = {}
            ignorable_input_cols = [x for x in ns.input_cols if (not ns.tss.is_timeseries or
                                                                 (x != ns.tss.order_by and
                                                                  x not in ns.tss.historical_columns))]
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
            acc_increases = []
            for col in ignorable_input_cols:
                accuracy_increase = (info['normal_accuracy'] - empty_input_accuracy[col])
                acc_increases.append(accuracy_increase)

            # low 0.2 temperature to accentuate differences
            acc_increases = t_softmax(torch.Tensor([acc_increases]), t=0.2).tolist()[0]
            for col, inc in zip(ignorable_input_cols, acc_increases):
                column_importances[col] = inc  # scores go from 0 to 1

            info['column_importances'] = column_importances

        return info
