from typing import Iterable


def can_be_nan_numeric(value: object) -> bool:
    """
    Tells us if **value** might be nan or inf or some other numeric value\
        (i.e. which can be cast as `float`) that is not actually a number.\
        Name is vague because I'm not 100% sure of all the edge cases of numeric\
        values that have number-like type behavior.
    """

    try:
        value = str(value)
        value = float(value)
    except Exception:
        return False

    try:
        if isinstance(value, float):
            a = int(value) # noqa
        isnan = False
    except Exception:
        isnan = True
    return isnan


def filter_nan_and_none(series: Iterable) -> list:
    return [x for x in series if not can_be_nan_numeric(x) and x is not None]
