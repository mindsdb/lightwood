import math
import importlib
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, mean_absolute_error


def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_functions: List[str]) -> float:
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        print(accuracy_function_str)
        if accuracy_function_str == 'evaluate_array_accuracy':
            nr_predictions = len(predictions.iloc[0])
            cols = [target] + [f'{target}_timestep_{i}' for i in range(1, nr_predictions)]
            true_values = data[cols].values.tolist()
            accuracy_function = evaluate_array_accuracy
        else:
            true_values = data[target].tolist()
            accuracy_function = getattr(importlib.import_module('sklearn.metrics'), accuracy_function_str)

        print(list(true_values), list(predictions), accuracy_function)
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


def evaluate_array_accuracy(
        true_values: List[List[Union[int, float]]],
        predictions: List[List[Union[int, float]]],
        **kwargs
) -> float:

    aggregate = 0
    nr_predictions = len(predictions[0])

    for i in range(len(predictions)):
        try:
            valid_horizon = [math.isnan(x) for x in true_values[i]].index(True)
        except ValueError:
            valid_horizon = len(true_values[i])

        aggregate += mean_absolute_error(predictions[i][:valid_horizon],
                                         true_values[i][:valid_horizon])

    return aggregate / len(predictions)
