from sklearn.metrics import r2_score as sk_r2_score
import numpy as np


def r2_score(y_true, y_pred) -> float:
    """ Wrapper for sklearn R2 score, lower capped between 0 and 1"""
    acc = sk_r2_score(y_true, y_pred)
    # Cap at 0
    if acc < 0:
        acc = 0
    # Guard against overflow (> 1 means overflow of negative score)
    if acc > 1:
        acc = 0

    return acc
