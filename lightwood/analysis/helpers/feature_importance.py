from typing import List
from copy import deepcopy

import torch
import numpy as np

from lightwood.ensemble import BaseEnsemble
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.general import evaluate_accuracy
from lightwood.analysis.nc.util import t_softmax


def compute_global_importance(
        predictor: BaseEnsemble,
        input_cols,
        target: str,
        val_data,
        encoded_data,
        normal_accuracy: float,
        accuracy_functions: List,
        is_classification: bool,
        ts_cfg: TimeseriesSettings
) -> dict:

    empty_input_accuracy = {}
    ignorable_input_cols = [x for x in input_cols if (not ts_cfg.is_timeseries or
                                                      (x not in ts_cfg.order_by and
                                                       x not in ts_cfg.historical_columns))]
    for col in ignorable_input_cols:
        partial_data = deepcopy(encoded_data)
        partial_data.clear_cache()
        for ds in partial_data.encoded_ds_arr:
            ds.data_frame[col] = [None] * len(ds.data_frame[col])

        if not is_classification:
            empty_input_preds = predictor(partial_data)
        else:
            empty_input_preds = predictor(partial_data, predict_proba=True)

        empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
            val_data,
            empty_input_preds['prediction'],
            target,
            accuracy_functions
        ).values()))

    column_importances = {}
    acc_increases = []
    for col in ignorable_input_cols:
        accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
        acc_increases.append(accuracy_increase)

    # low 0.2 temperature to accentuate differences
    acc_increases = t_softmax(torch.Tensor([acc_increases]), t=0.2).tolist()[0]
    for col, inc in zip(ignorable_input_cols, acc_increases):
        column_importances[col] = 10 * inc  # scores go from 0 to 10 in GUI

    return column_importances
