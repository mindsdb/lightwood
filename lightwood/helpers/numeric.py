from typing import Iterable
from type_infer.helpers import is_nan_numeric


def filter_nan_and_none(series: Iterable) -> list:
    return [x for x in series if not is_nan_numeric(x) and x is not None]
