import importlib
from typing import List, Union, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, mean_absolute_error
from lightwood.helpers.numeric import is_nan_numeric


def get_group_matches(data, combination):
    """Given a grouped-by combination, return rows of the data that match belong to it. Params:
    data: dict with data to filter and group-by columns info.
    combination: tuple with values to filter by
    return: indexes for rows to normalize, data to normalize
    """
    keys = data['group_info'].keys()  # which column does each combination value belong to

    if isinstance(data['data'], pd.Series):
        data['data'] = np.vstack(data['data'])
    if isinstance(data['data'], np.ndarray) and len(data['data'].shape) < 2:
        data['data'] = np.expand_dims(data['data'], axis=1)

    if combination == '__default':
        idxs = range(len(data['data']))
        return [idxs, np.array(data['data'])[idxs, :]]  # return all data
    else:
        all_sets = []
        for val, key in zip(combination, keys):
            all_sets.append(set([i for i, elt in enumerate(data['group_info'][key]) if elt == val]))
        if all_sets:
            idxs = list(set.intersection(*all_sets))
            return idxs, np.array(data['data'])[idxs, :]

        else:
            return [], np.array([])


# ------------------------- #
# Accuracy metrics
# ------------------------- #
def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_functions: List[str],
                      ts_analysis: Optional[dict] = {}) -> Dict[str, float]:
    """
    Dispatcher for accuracy evaluation.
    
    :param data: original dataframe.
    :param predictions: output of a lightwood predictor for the input `data`.
    :param target: target column name.
    :param accuracy_functions: list of accuracy function names. Support currently exists for `scikit-learn`'s `metrics` module, plus any custom methods that Lightwood exposes.
    :param ts_analysis: `lightwood.data.timeseries_analyzer` output, used to compute time series task accuracy.
    :return: accuracy metric for a dataset and predictions.
    """  # noqa
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        if accuracy_function_str == 'evaluate_array_accuracy':
            nr_predictions = 1 if not isinstance(predictions.iloc[0], list) else len(predictions.iloc[0])
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
    """
    Evaluates accuracy for regression tasks.
    If predictions have a lower and upper bound, then `within-bound` accuracy is computed: whether the ground truth value falls within the predicted region.
    If not, then a (positive bounded) R2 score is returned instead.
    
    :return: accuracy score as defined above. 
    """  # noqa
    if 'lower' and 'upper' in predictions:
        Y = np.array(true_values).astype(float)
        within = ((Y >= predictions['lower']) & (Y <= predictions['upper']))
        return sum(within) / len(within)
    else:
        r2 = r2_score(true_values, predictions['prediction'])
        return max(r2, 0)


def evaluate_multilabel_accuracy(true_values, predictions, **kwargs):
    """
    Evaluates accuracy for multilabel/tag prediction.

    :return: weighted f1 score of predictions and ground truths.
    """
    pred_values = predictions['prediction']
    return f1_score(true_values, pred_values, average='weighted')


def evaluate_array_accuracy(
        true_values: List[List[Union[int, float]]],
        predictions: List[List[Union[int, float]]],
        data: pd.DataFrame,
        **kwargs
) -> float:
    """
    Evaluate accuracy in numerical time series forecasting tasks.
    Defaults to mean absolute scaled error (MASE) if in-sample residuals are available.
    If this is not the case, R2 score is computed instead.

    Scores are computed for each timestep (as determined by `timeseries_settings.nr_predictions`),
    and the final accuracy is the reciprocal of the average score through all timesteps.
    """

    ts_analysis = kwargs.get('ts_analysis', {})
    naive_errors = ts_analysis.get('ts_naive_mae', {})

    if not naive_errors:
        # use mean R2 method if naive errors are not available
        return evaluate_array_r2_accuracy(true_values, predictions, ts_analysis=ts_analysis)

    mases = []
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    wrapped_data = {'data': data.reset_index(drop=True),
                    'group_info': {gcol: data[gcol].tolist()
                                   for gcol in ts_analysis['tss'].group_by} if ts_analysis['tss'].group_by else {}
                    }
    for group in ts_analysis['group_combinations']:
        g_idxs, _ = get_group_matches(wrapped_data, group)

        # only evaluate populated groups
        if g_idxs:
            trues = true_values[g_idxs]
            preds = predictions[g_idxs]

            if ts_analysis['tss'].nr_predictions == 1:
                preds = np.expand_dims(preds, axis=1)

            # only evaluate accuracy for rows with complete historical context
            if len(trues) > ts_analysis['tss'].window:
                trues = trues[ts_analysis['tss'].window:]
                preds = preds[ts_analysis['tss'].window:]

            # add MASE score for each group (__default only considered if the task is non-grouped)
            if len(ts_analysis['group_combinations']) == 1 or group != '__default':
                mases.append(mase(trues, preds, ts_analysis['ts_naive_mae'][group], ts_analysis['tss'].nr_predictions))

    return 1 / max(np.average(mases), 1e-4)  # reciprocal to respect "larger -> better" convention


def evaluate_array_r2_accuracy(
        true_values: List[List[Union[int, float]]],
        predictions: List[List[Union[int, float]]],
        **kwargs
) -> float:
    """
    Default time series forecasting accuracy method.
    Returns mean R2 score over all timesteps in the forecasting horizon.
    """
    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))

    aggregate = 0.0

    fh = 1 if not isinstance(predictions[0], list) else len(predictions[0])
    if fh == 1:
        predictions = [[p] for p in predictions]

    # only evaluate accuracy for rows with complete historical context
    if kwargs.get('ts_analysis', {}).get('tss', False):
        true_values = true_values[kwargs['ts_analysis']['tss'].window:]
        predictions = predictions[kwargs['ts_analysis']['tss'].window:]

    for i in range(fh):
        aggregate += base_acc_fn([t[i] for t in true_values], [p[i] for p in predictions])

    return aggregate / fh


# ------------------------- #
# Helpers
# ------------------------- #
def mase(trues, preds, scale_error, fh):
    """
    Computes mean absolute scaled error.
    The scale corrective factor is the mean in-sample residual from the naive forecasting method.
    """
    if scale_error == 0:
        scale_error = 1  # cover (rare) case where series is constant

    agg = 0.0
    for i in range(fh):
        true = [t[i] for t in trues]
        pred = [p[i] for p in preds]
        agg += mean_absolute_error(true, pred)

    return (agg / fh) / scale_error


def is_none(value):
    """
    We use pandas :(
    Pandas has no way to guarantee "stability" for the type of a column, it choses to arbitrarily change it based on the values.
    Pandas also change the values in the columns based on the types.
    Lightwood relies on having ``None`` values for a cells that represent "missing" or "corrupt".
    
    When we assign ``None`` to a cell in a dataframe this might get turned to `nan` or other values, this function checks if a cell is ``None`` or any other values a pd.DataFrame might convert ``None`` to.

    It also check some extra values (like ``''``) that pandas never converts ``None`` to (hopefully). But lightwood would still consider those values "None values", and this will allow for more generic use later.
    """ # noqa
    if value is None:
        return True

    if is_nan_numeric(value):
        return True

    if str(value) == '':
        return True

    if str(value) in ('None', 'nan', 'NaN', 'np.nan'):
        return True

    return False

