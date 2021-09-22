import re
from copy import deepcopy

import pandas as pd
import datetime
from dateutil.parser import parse as parse_dt

from lightwood.api.dtype import dtype
from lightwood.helpers.text import clean_float
from lightwood.helpers.log import log
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.numeric import can_be_nan_numeric

from typing import Dict, List, Optional, Tuple, Callable


def cleaner(
    data: pd.DataFrame,
    dtype_dict: Dict[str, str],
    pct_invalid: float,
    ignore_features: List[str],
    identifiers: Dict[str, str],
    target: str,
    mode: str,
    timeseries_settings: TimeseriesSettings,
    anomaly_detection: bool,
) -> pd.DataFrame:

    log.info("My cleaner deployed!")
    # Drop columns we don't want to use
    data = deepcopy(data)
    to_drop = [*ignore_features, [x for x in identifiers.keys() if x != target]]
    exceptions = ["__mdb_make_predictions"]
    for col in to_drop:
        try:
            data = data.drop(columns=[col])
        except Exception:
            pass

    if mode == "train":
        data = clean_empty_targets(data, target)
    if mode == "predict":
        if (
            target in data.columns
            and not timeseries_settings.use_previous_target
            and not anomaly_detection
        ):
            data = data.drop(columns=[target])

    # Drop extra columns
    for name in list(data.columns):
        if name not in dtype_dict and name not in exceptions:
            data = data.drop(columns=[name])

    # Standardize content
    for name, data_dtype in dtype_dict.items():
        if mode == "predict":
            if name == target:
                continue
        if name in to_drop:
            continue
        if name not in data.columns:
            if "__mdb_ts_previous" not in name:
                data[name] = [None] * len(data)
            continue

        # Gets cleaning function and applies to data
        clean_fxn = get_cleaning_func(data_dtype)

        data[name] = data[name].apply(clean_fxn)

        if check_invalid(data[name], pct_invalid):
            err = f"Too many ({pct_invalid}%) invalid values in column {name} of type {data_dtype}"
            log.error(err)
            raise Exception(err)

    return data


def check_invalid(new_data: pd.Series, pct_invalid: float) -> bool:
    """ Checks how many invalid data points there are """

    chk_invalid = (
        100
        * (len(new_data) - len([x for x in new_data if x is not None]))
        / len(new_data)
    )

    return chk_invalid > pct_invalid


def get_cleaning_func(data_dtype: dtype) -> Callable:
    """
    For the provided data type, provide the appropriate cleaning function. Below are the defaults, users can either override this function OR impose a custom block.

    :param data_dtype: The data-type (inferred from a column) as prescribed from ``api.dtype``

    :returns: The appropriate function that will pre-process (clean) data of specified dtype.
    """
    if data_dtype in (dtype.date, dtype.datetime):
        clean_func = _standardize_datetime

    elif data_dtype in (dtype.float):
        clean_func = _clean_numeric

    elif data_dtype in (dtype.integer):
        clean_fun = (
            lambda x: int(_clean_numeric(x)) if _clean_numeric(x) is not None else None
        )

    elif data_dtype in (dtype.array):
        clean_func = _standardize_array

    elif data_dtype in (dtype.tags):
        clean_func = _tags_to_tuples

    elif data_dtype in (dtype.quantity):
        clean_func = lambda x: float(re.sub("[^0-9.,]", "", x).replace(",", "."))

    elif data_dtype in (
        dtype.short_text,
        dtype.rich_text,
        dtype.categorical,
        dtype.binary,
    ):
        clean_func = lambda x: str(x)

    else:
        raise ValueError(f"{data_dtype} is not supported. Check lightwood.api.dtype")

    return clean_func


# ------------------------- #
# Temporal Cleaning
# ------------------------- #


def _standardize_datetime(element: object) -> Optional[float]:
    """
    Parses an expected date-time element. Intakes an element that can in theory be anything.
    """
    try:
        date = parse_dt(str(element))
    except Exception:
        try:
            date = datetime.datetime.utcfromtimestamp(element)
        except Exception:
            return None

    return date.timestamp()


# ------------------------- #
# Tags/Sequences
# ------------------------- #

# TODO Make it split on something other than commas
def _tags_to_tuples(tags_str: str) -> Tuple[str]:
    """
    Converts comma-separated values into a tuple to preserve a sequence/array.

    Ex:
    >> x = 'apples, oranges, bananas'
    >> _tags_to_tuples(x)
    >> ('apples', 'oranges', 'bananas')
    """
    try:
        return tuple([x.strip() for x in tags_str.split(",")])
    except Exception:
        return tuple()


def _standardize_array(element: object) -> Optional[Union[List[float], float]]:
    """
    Given an array of numbers in the form ``[1, 2, 3, 4]``, converts into a numerical sequence.

    :param element: An array-like element in a sequence
    :returns: standardized array OR scalar number IF edge case

    Ex of edge case:
    >> element = [1]
    >> _standardize_array(element)
    >> 1
    """
    try:
        element = str(element)
        element = element.rstrip("]").lstrip("[")
        element = element.rstrip(" ").lstrip(" ")
        element = element.replace(", ", " ").replace(",", " ")
        # Handles cases where arrays are numbers
        if " " not in element:
            element = _clean_numeric(element)
        else:
            element = [float(x) for x in element.split(" ")]
    except Exception:
        pass

    return element


# ------------------------- #
# Numeric
# ------------------------- #


def _clean_numeric(element: object) -> Optional[float]:
    """
    Given an element, converts it into a numeric format. If element is NaN, or inf, then returns None.
    """
    try:
        cleaned_float = clean_float(element)
        if can_be_nan_numeric(cleaned_float):
            return None
        return cleaned_float
    except Exception:
        return None


# ----------------- #
# Empty/Missing/NaN handling
# ----------------- #


def clean_empty_targets(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Drop any rows that have targets as unknown. Targets are necessary to train.

    :param df: The input dataframe including the target value
    :param target: the column name that is the output target variable

    :returns: Data with any target smissing
    """
    # Compare length before/after
    len_before = len(df)

    # Use Pandas ```dropna``` to omit any rows with missing values for targets; these cannot be trained
    df = df.dropna(subset=[target])

    # Compare length with after
    len_after = len(df)
    nr_removed = len_before - len_after

    if nr_removed != 0:
        log.warning(
            f"Removed {nr_removed} rows because target was missing. Training on these rows is not possible."
        )  # noqa

    return df
