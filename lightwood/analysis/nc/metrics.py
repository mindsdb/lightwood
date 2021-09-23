# Original author: Henrik Linusson (github.com/donlnz)

import numpy as np


# -----------------------------------------------------------------------------
# Validity measures
# -----------------------------------------------------------------------------
def reg_n_correct(prediction, y, significance=None):
    """Calculates the number of correct predictions made by a conformal
    regression model.
    """
    if significance is not None:
        idx = int(significance * 100 - 1)
        prediction = prediction[:, :, idx]

    low = y >= prediction[:, 0]
    high = y <= prediction[:, 1]
    correct = low * high

    return y[correct].size


def reg_mean_errors(prediction, y, significance):
    """Calculates the average error rate of a conformal regression model.
    """
    return 1 - reg_n_correct(prediction, y, significance) / y.size


def class_n_correct(prediction, y, significance):
    """Calculates the number of correct predictions made by a conformal
    classification model.
    """
    labels, y = np.unique(y, return_inverse=True)
    prediction = prediction > significance
    correct = np.zeros((y.size,), dtype=bool)
    for i, y_ in enumerate(y):
        correct[i] = prediction[i, int(y_)]
    return np.sum(correct)


def class_mean_errors(prediction, y, significance=None):
    """Calculates the average error rate of a conformal classification model.
    """
    return 1 - (class_n_correct(prediction, y, significance) / y.size)


def class_one_err(prediction, y, significance=None):
    """Calculates the error rate of conformal classifier predictions containing
     only a single output label.
    """
    labels, y = np.unique(y, return_inverse=True)
    prediction = prediction > significance
    idx = np.arange(0, y.size, 1)
    idx = filter(lambda x: np.sum(prediction[x, :]) == 1, idx)
    errors = filter(lambda x: not prediction[x, int(y[x])], idx)

    if len(idx) > 0:
        return np.size(errors) / np.size(idx)
    else:
        return 0


def class_mean_errors_one_class(prediction, y, significance, c=0):
    """Calculates the average error rate of a conformal classification model,
      considering only test examples belonging to class ``c``. Use
      ``functools.partial`` in order to test other classes.
    """
    labels, y = np.unique(y, return_inverse=True)
    prediction = prediction > significance
    idx = np.arange(0, y.size, 1)[y == c]
    errs = np.sum(1 for _ in filter(lambda x: not prediction[x, c], idx))

    if idx.size > 0:
        return errs / idx.size
    else:
        return 0


def class_one_err_one_class(prediction, y, significance, c=0):
    """Calculates the error rate of conformal classifier predictions containing
     only a single output label. Considers only test examples belonging to
     class ``c``. Use ``functools.partial`` in order to test other classes.
    """
    labels, y = np.unique(y, return_inverse=True)
    prediction = prediction > significance
    idx = np.arange(0, y.size, 1)
    idx = filter(lambda x: prediction[x, c], idx)
    idx = filter(lambda x: np.sum(prediction[x, :]) == 1, idx)
    errors = filter(lambda x: int(y[x]) != c, idx)

    if len(idx) > 0:
        return np.size(errors) / np.size(idx)
    else:
        return 0


# -----------------------------------------------------------------------------
# Efficiency measures
# -----------------------------------------------------------------------------
def _reg_interval_size(prediction, y, significance):
    idx = int(significance * 100 - 1)
    prediction = prediction[:, :, idx]

    return prediction[:, 1] - prediction[:, 0]


def reg_min_size(prediction, y, significance):
    return np.min(_reg_interval_size(prediction, y, significance))


def reg_q1_size(prediction, y, significance):
    return np.percentile(_reg_interval_size(prediction, y, significance), 25)


def reg_median_size(prediction, y, significance):
    return np.median(_reg_interval_size(prediction, y, significance))


def reg_q3_size(prediction, y, significance):
    return np.percentile(_reg_interval_size(prediction, y, significance), 75)


def reg_max_size(prediction, y, significance):
    return np.max(_reg_interval_size(prediction, y, significance))


def reg_mean_size(prediction, y, significance):
    """Calculates the average prediction interval size of a conformal
    regression model.
    """
    return np.mean(_reg_interval_size(prediction, y, significance))


def class_avg_c(prediction, y, significance):
    """Calculates the average number of classes per prediction of a conformal
    classification model.
    """
    prediction = prediction > significance
    return np.sum(prediction) / prediction.shape[0]


def class_mean_p_val(prediction, y, significance):
    """Calculates the mean of the p-values output by a conformal classification
    model.
    """
    return np.mean(prediction)


def class_one_c(prediction, y, significance):
    """Calculates the rate of singleton predictions (prediction sets containing
    only a single class label) of a conformal classification model.
    """
    prediction = prediction > significance
    n_singletons = np.sum(1 for _ in filter(lambda x: np.sum(x) == 1, prediction))
    return n_singletons / y.size


def class_empty(prediction, y, significance):
    """Calculates the rate of singleton predictions (prediction sets containing
    only a single class label) of a conformal classification model.
    """
    prediction = prediction > significance
    n_empty = np.sum(1 for _ in filter(lambda x: np.sum(x) == 0, prediction))
    return n_empty / y.size


def n_test(prediction, y, significance):
    """Provides the number of test patters used in the evaluation.
    """
    return y.size
