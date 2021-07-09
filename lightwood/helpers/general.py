from lightwood.api import predictor
from typing import List
from sklearn.metrics import r2_score, f1_score, balanced_accuracy_score
import importlib
import numpy as np
import pandas as pd


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
        return sum(within)/len(within)
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


def evaluate_array_accuracy(true_values, predictions, **kwargs):
    accuracy = 0
    predictions = predictions['prediction']
    true_values = list(true_values)
    acc_f = balanced_accuracy_score if kwargs['categorical'] else r2_score
    for i in range(len(predictions)):
        if isinstance(true_values[i], list):
            accuracy += max(0, acc_f(predictions[i], true_values[i]))
        else:
            # For the T+1 usecase
            return max(0, acc_f([x[0] for x in predictions], true_values))
    return accuracy / len(predictions)

