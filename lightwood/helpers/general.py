import importlib
from copy import deepcopy
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score, mean_absolute_error, balanced_accuracy_score
from type_infer.helpers import is_nan_numeric
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from lightwood.helpers.ts import get_group_matches


# ------------------------- #
# Accuracy metrics
# ------------------------- #
def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_functions: List[str],
                      ts_analysis: Optional[dict] = {},
                      n_decimals: Optional[int] = 3) -> Dict[str, float]:
    """
    Dispatcher for accuracy evaluation.
    
    :param data: original dataframe.
    :param predictions: output of a lightwood predictor for the input `data`.
    :param target: target column name.
    :param accuracy_functions: list of accuracy function names. Support currently exists for `scikit-learn`'s `metrics` module, plus any custom methods that Lightwood exposes.
    :param ts_analysis: `lightwood.data.timeseries_analyzer` output, used to compute time series task accuracy.
    :param n_decimals: used to round accuracies.
    :return: accuracy metric for a dataset and predictions.
    """  # noqa
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        if 'array_accuracy' in accuracy_function_str or accuracy_function_str in ('bounded_ts_accuracy', ):
            if ts_analysis is None or not ts_analysis.get('tss', False) or not ts_analysis['tss'].is_timeseries:
                # normal array, needs to be expanded
                cols = [target]
                true_values = data[cols].apply(lambda x: pd.Series(x[target]), axis=1)
            else:
                horizon = 1 if not isinstance(predictions.iloc[0], list) else len(predictions.iloc[0])
                gby = ts_analysis.get('tss', {}).group_by if ts_analysis.get('tss', {}).group_by else []
                cols = [target] + [f'{target}_timestep_{i}' for i in range(1, horizon)] + gby
                true_values = data[cols]

            true_values = true_values.reset_index(drop=True)
            predictions = predictions.apply(pd.Series).reset_index(drop=True)  # split array cells into columns

            if accuracy_function_str == 'evaluate_array_accuracy':
                acc_fn = evaluate_array_accuracy
            elif accuracy_function_str == 'evaluate_num_array_accuracy':
                acc_fn = evaluate_num_array_accuracy
            elif accuracy_function_str == 'evaluate_cat_array_accuracy':
                acc_fn = evaluate_cat_array_accuracy
            elif accuracy_function_str == 'complementary_smape_array_accuracy':
                acc_fn = complementary_smape_array_accuracy
            else:
                acc_fn = bounded_ts_accuracy

            score_dict[accuracy_function_str] = acc_fn(true_values,
                                                       predictions,
                                                       data=data[cols],
                                                       ts_analysis=ts_analysis)
        else:
            true_values = data[target].tolist()
            if hasattr(importlib.import_module('lightwood.helpers.accuracy'), accuracy_function_str):
                accuracy_function = getattr(importlib.import_module('lightwood.helpers.accuracy'),
                                            accuracy_function_str)
            else:
                accuracy_function = getattr(importlib.import_module('sklearn.metrics'), accuracy_function_str)
            score_dict[accuracy_function_str] = accuracy_function(list(true_values), list(predictions))

    for fn, score in score_dict.items():
        score_dict[fn] = round(score, n_decimals)

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


def evaluate_num_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    Evaluate accuracy in numerical time series forecasting tasks.
    Defaults to mean absolute scaled error (MASE) if in-sample residuals are available.
    If this is not the case, R2 score is computed instead.

    Scores are computed for each timestep (as determined by `timeseries_settings.horizon`),
    and the final accuracy is the reciprocal of the average score through all timesteps.
    """
    def _naive(true_values, predictions):
        nan_mask = (~np.isnan(true_values)).astype(int)
        predictions *= nan_mask
        true_values = np.nan_to_num(true_values, 0.0)
        return evaluate_array_accuracy(true_values, predictions, ts_analysis=ts_analysis)

    ts_analysis = kwargs.get('ts_analysis', {})
    if not ts_analysis:
        naive_errors = None
    else:
        naive_errors = ts_analysis.get('ts_naive_mae', {})
        if ts_analysis['tss'].group_by:
            [true_values.pop(gby_col) for gby_col in ts_analysis['tss'].group_by]

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    if not naive_errors:
        # use mean R2 method if naive errors are not available
        return _naive(true_values, predictions)

    mases = []
    for group in ts_analysis['group_combinations']:
        g_idxs, _ = get_group_matches(kwargs['data'].reset_index(drop=True), group, ts_analysis['tss'].group_by)

        # only evaluate populated groups
        if g_idxs:
            trues = true_values[g_idxs]
            preds = predictions[g_idxs]

            # add MASE score for each group (__default only considered if the task is non-grouped)
            if len(ts_analysis['group_combinations']) == 1 or group != '__default':
                try:
                    mases.append(mase(trues, preds, ts_analysis['ts_naive_mae'][group], ts_analysis['tss'].horizon))
                except Exception:
                    # group is novel, ignore for accuracy reporting purposes
                    pass

    acc = 1 / max(np.average(mases), 1e-4)  # reciprocal to respect "larger -> better" convention
    if acc != acc:
        return _naive(true_values, predictions)  # nan due to having only novel groups in validation, forces reversal
    else:
        return acc


def evaluate_array_accuracy(
        true_values: np.ndarray,
        predictions: np.ndarray,
        **kwargs
) -> float:
    """
    Default time series forecasting accuracy method.
    Returns mean score over all timesteps in the forecasting horizon, as determined by the `base_acc_fn` (R2 score by default).
    """  # noqa
    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))

    fh = true_values.shape[1]

    aggregate = 0.0
    for i in range(fh):
        aggregate += base_acc_fn([t[i] for t in true_values], [p[i] for p in predictions])

    return aggregate / fh


def evaluate_cat_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    Evaluate accuracy in categorical time series forecasting tasks.

    Balanced accuracy is computed for each timestep (as determined by `timeseries_settings.horizon`),
    and the final accuracy is the reciprocal of the average score through all timesteps.
    """
    ts_analysis = kwargs.get('ts_analysis', {})

    if ts_analysis and ts_analysis['tss'].group_by:
        [true_values.pop(gby_col) for gby_col in ts_analysis['tss'].group_by]

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    return evaluate_array_accuracy(true_values,
                                   predictions,
                                   ts_analysis=ts_analysis,
                                   base_acc_fn=balanced_accuracy_score)


def bounded_ts_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    The normal MASE accuracy inside ``evaluate_array_accuracy`` has a break point of 1.0: smaller values mean a naive forecast is better, and bigger values imply the forecast is better than a naive one. It is upper-bounded by 1e4.

    This 0-1 bounded MASE variant scores the 1.0 breakpoint to be equal to 0.5.
    For worse-than-naive, it scales linearly (with a factor).
    For better-than-naive, we fix 10 as 0.99, and scaled-logarithms (with 10 and 1e4 cutoffs as respective bases) are used to squash all remaining preimages to values between 0.5 and 1.0.
    """  # noqa
    true_values = deepcopy(true_values)
    predictions = deepcopy(predictions)
    result = evaluate_num_array_accuracy(true_values,
                                         predictions,
                                         **kwargs)
    sp = 5
    if sp < result <= 1e4:
        step_base = 0.99
        return step_base + (np.log(result) / np.log(1e4)) * (1 - step_base)
    elif 1 <= result <= sp:
        step_base = 0.5
        return step_base + (np.log(result) / np.log(sp)) * (0.99 - step_base)
    else:
        return result / 2  # worse than naive


def complementary_smape_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    This metric is used in forecasting tasks. It returns ``1 - (sMAPE/2)``, where ``sMAPE`` is the symmetrical mean absolute percentage error of the forecast versus actual measurements in the time series.

    As such, its domain is 0-1 bounded.
    """  # noqa
    y_true = deepcopy(true_values)
    y_pred = deepcopy(predictions)
    tss = kwargs.get('ts_analysis', {}).get('tss', False)
    if tss and tss.group_by:
        [y_true.pop(gby_col) for gby_col in kwargs['ts_analysis']['tss'].group_by]

    # nan check
    y_true = y_true.values
    y_pred = y_pred.values
    if np.isnan(y_true).any():
        # convert all nan indexes to zero pairs that don't contribute to the metric
        nans = np.isnan(y_true)
        y_true[nans] = 0
        y_pred[nans] = 0

    smape_score = mean_absolute_percentage_error(y_true, y_pred, symmetric=True)
    return 1 - smape_score / 2


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

    nan_mask = (~np.isnan(trues)).astype(int)
    preds *= nan_mask
    trues = np.nan_to_num(trues, 0.0)

    agg = 0.0
    for i in range(fh):
        true = trues[:, i]
        pred = preds[:, i]
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

