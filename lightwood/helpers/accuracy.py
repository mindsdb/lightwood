from typing import Iterable, List
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.metrics import precision_score as sk_precision_score


def to_binary(y: Iterable) -> List[int]:
    try:
        y_binarized = []
        for ele in y:
            if str(ele).lower() == 'true':
                y_binarized.append(1)
            elif str(ele).lower() == 'false':
                y_binarized.append(0)
            else:
                y_binarized.append(int(ele))

        assert len(set(y_binarized)) < 3
        assert 1 in y_binarized or 0 in y_binarized
    except Exception:
        raise Exception('To use precision, recall or f1 please make sure your target consists only of 1s and 0s')
    return y_binarized


def f1_score(y_true, y_pred) -> float:
    return sk_f1_score(to_binary(y_true), to_binary(y_pred))


def recall_score(y_true, y_pred) -> float:
    return sk_recall_score(to_binary(y_true), to_binary(y_pred))


def precision_score(y_true, y_pred) -> float:
    return sk_precision_score(to_binary(y_true), to_binary(y_pred))


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
