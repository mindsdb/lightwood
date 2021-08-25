import math
import importlib
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score


def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_functions: List[str]) -> Dict[str, float]:
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        if accuracy_function_str == 'evaluate_array_accuracy':
            nr_predictions = len(predictions.iloc[0])
            cols = [target] + [f'{target}_timestep_{i}' for i in range(1, nr_predictions)]
            true_values = data[cols].values.tolist()
            accuracy_function = evaluate_array_accuracy
        else:
            true_values = data[target].tolist()
            accuracy_function = getattr(importlib.import_module('sklearn.metrics'), accuracy_function_str)

        score_dict[accuracy_function_str] = accuracy_function(list(true_values), list(predictions))

    return score_dict


def evaluate_regression_accuracy(
        true_values,
        predictions,
        **kwargs
):
    if 'lower' and 'upper' in predictions:
        Y = np.array(true_values).astype(float)
        within = ((Y >= predictions['lower']) & (Y <= predictions['upper']))
        return sum(within) / len(within)
    else:
        r2 = r2_score(true_values, predictions['prediction'])
        return max(r2, 0)


def evaluate_multilabel_accuracy(true_values, predictions, **kwargs):
    pred_values = predictions['prediction']
    return f1_score(true_values, pred_values, average='weighted')


def evaluate_array_accuracy(
        true_values: List[List[Union[int, float]]],
        predictions: List[List[Union[int, float]]],
        **kwargs
) -> float:
    # @TODO: ideally MASE here
    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))
    aggregate = 0

    for i in range(len(predictions)):
        try:
            valid_horizon = [math.isnan(x) for x in true_values[i]].index(True)
        except ValueError:
            valid_horizon = len(true_values[i])

        aggregate += base_acc_fn(true_values[i][:valid_horizon], predictions[i][:valid_horizon])

    return aggregate / len(predictions)
