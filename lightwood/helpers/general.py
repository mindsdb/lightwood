import importlib
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, mean_absolute_error


from lightwood.encoder.time_series.helpers.common import get_group_matches


def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_functions: List[str],
                      ts_analysis: Optional[dict] = None) -> Dict[str, float]:
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        if accuracy_function_str == 'evaluate_array_accuracy':
            nr_predictions = len(predictions.iloc[0])
            cols = [target] + [f'{target}_timestep_{i}' for i in range(1, nr_predictions)]
            true_values = data[cols].values.tolist()
            score_dict[accuracy_function_str] = evaluate_array_accuracy(list(true_values),
                                                                        list(predictions),
                                                                        data,
                                                                        ts_analysis=ts_analysis)
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
        data: pd.DataFrame,
        **kwargs
) -> float:
    def mase(trues, preds, scale_error, fh):
        agg = 0
        for i in range(fh):
            true = [t[i] for t in trues]
            pred = [p[i] for p in preds]
            agg += mean_absolute_error(true, pred)

        return agg / scale_error

    ts_analysis = kwargs.get('ts_analysis', {})
    if not ts_analysis:
        # use mean R2 method if naive errors were not computed
        return evaluate_array_r2_accuracy(true_values, predictions)
    else:
        true_values = np.array(true_values)
        predictions = np.array(predictions)
        mases = []
        wrapped_data = {'data': data.reset_index(drop=True),
                        'group_info': {gcol: data[gcol].tolist()
                                       for gcol in ts_analysis['tss'].group_by} if ts_analysis['tss'].group_by else {}
                        }
        for group in ts_analysis['group_combinations']:
            g_idxs, _ = get_group_matches(wrapped_data, group)
            trues = true_values[g_idxs]
            preds = predictions[g_idxs]

            # add MASE score for each group (__default only considered if the task is non-grouped)
            if len(ts_analysis['group_combinations']) == 1 or group != '__default':
                mases.append(mase(trues, preds, ts_analysis['ts_naive_mae'][group], ts_analysis['tss'].nr_predictions))

    return 1 / max(np.average(mases), 1e-3)  # reciprocal to respect "larger -> better" convention


def evaluate_array_r2_accuracy(
        true_values: List[List[Union[int, float]]],
        predictions: List[List[Union[int, float]]],
        **kwargs
) -> float:
    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))

    aggregate = 0
    fh = len(predictions[0])

    for i in range(fh):
        aggregate += base_acc_fn([t[i] for t in true_values], [p[i] for p in predictions])

    return aggregate / fh
