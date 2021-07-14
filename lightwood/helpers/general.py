import importlib
from typing import List, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import r2_score, f1_score, mean_absolute_error


def evaluate_accuracy(true_values: pd.Series,
                      predictions: pd.Series,
                      accuracy_functions: List[str]) -> float:
    score_dict = {}
    for accuracy_function_str in accuracy_functions:
        if accuracy_function_str == 'evaluate_array_accuracy':
            accuracy_function = evaluate_array_accuracy
        else:
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


# This is ugly because it uses the encoders, figure out how to do it without them
# Maybe literally one-hot encode inside the accuracy functions
def evaluate_multilabel_accuracy(column, predictions, true_values, backend, **kwargs):
    # @TODO: use new API
    encoder = backend.predictor._mixer.encoders[column]
    pred_values = encoder.encode(predictions['prediction'])
    true_values = encoder.encode(true_values)
    return f1_score(true_values, pred_values, average='weighted')


def evaluate_array_accuracy(true_values: List[Union[int, float]], predictions: List[List[Union[int, float]]], **kwargs):
    nr_predictions = len(predictions) / len(true_values)  # @TODO: pass instead of inferring
    assert nr_predictions % 1 == 0

    formatted_predictions = np.array([predictions[i:i + int(nr_predictions)]
                                      for i in range(int(len(predictions) // nr_predictions))])
    formatted_truths = sliding_window_view(np.array(true_values + [np.nan for _ in range(int(nr_predictions - 1))]),
                                           int(nr_predictions)).copy()
    formatted_truths[np.isnan(formatted_truths)] = 0.0

    aggregate = 0
    for i in range(len(formatted_predictions)):
        aggregate += mean_absolute_error(formatted_predictions[i], formatted_truths[i])

    return aggregate / len(predictions)
