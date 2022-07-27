import re
import datetime
from copy import deepcopy
from dateutil.parser import parse as parse_dt
from typing import Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import pandas as pd

from lightwood.api.dtype import dtype
from lightwood.helpers import text
from lightwood.helpers.log import log
from lightwood.helpers.imputers import BaseImputer
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.numeric import is_nan_numeric


def cleaner(
    data: pd.DataFrame,
    dtype_dict: Dict[str, str],
    pct_invalid: float,
    identifiers: Dict[str, str],
    target: str,
    mode: str,
    timeseries_settings: TimeseriesSettings,
    anomaly_detection: bool,
    imputers: Dict[str, BaseImputer] = {},
    custom_cleaning_functions: Dict[str, str] = {}
) -> pd.DataFrame:
    """
    The cleaner is a function which takes in the raw data, plus additional information about it's types and about the problem. Based on this it generates a "clean" representation of the data, where each column has an ideal standardized type and all malformed or otherwise missing or invalid elements are turned into ``None``. Optionally, these ``None`` values can be replaced with imputers.

    :param data: The raw data
    :param dtype_dict: Type information for each column
    :param pct_invalid: How much of each column can be invalid
    :param identifiers: A dict containing all identifier typed columns
    :param target: The target columns
    :param mode: Can be "predict" or "train"
    :param imputers: The key corresponds to the single input column that will be imputed by the object. Refer to the imputer documentation for more details.
    :param timeseries_settings: Timeseries related settings, only relevant for timeseries predictors, otherwise can be the default object
    :param anomaly_detection: Are we detecting anomalies with this predictor?

    :returns: The cleaned data
    """ # noqa

    data = _remove_columns(data, identifiers, target, mode, timeseries_settings,
                           anomaly_detection, dtype_dict)

    data['__mdb_original_index'] = np.arange(len(data))

    for col in _get_columns_to_clean(data, dtype_dict, mode, target):

        # Get and apply a cleaning function for each data type
        # If you want to customize the cleaner, it's likely you can to modify ``get_cleaning_func``
        data[col] = data[col].apply(get_cleaning_func(dtype_dict[col], custom_cleaning_functions))

    if timeseries_settings.is_timeseries:
        data = clean_timeseries(data, timeseries_settings)

    for col, imputer in imputers.items():
        if col in data.columns:
            cols = [col] + [col for col in imputer.dependencies]
            data[col] = imputer.impute(data[cols])

    return data


def _check_if_invalid(new_data: pd.Series, pct_invalid: float, col_name: str):
    """
    Checks how many invalid data points there are. Invalid data points are flagged as "Nones" from the cleaning processs (see data/cleaner.py for default).
    If there are too many invalid data points (specified by `pct_invalid`), then an error message will pop up. This is used as a safeguard for very messy data.

    :param new_data: data to check for invalid values.
    :param pct_invalid: maximum percentage of invalid values. If this threshold is surpassed, an exception is raised.
    :param col_name: name of the column to analyze.

    """  # noqa

    chk_invalid = (
        100
        * (len(new_data) - len([x for x in new_data if x is not None]))
        / len(new_data)
    )

    if chk_invalid > pct_invalid:
        err = f'Too many ({chk_invalid}%) invalid values in column {col_name}nam'
        log.error(err)
        raise Exception(err)


def get_cleaning_func(data_dtype: dtype, custom_cleaning_functions: Dict[str, str]) -> Callable:
    """
    For the provided data type, provide the appropriate cleaning function. Below are the defaults, users can either override this function OR impose a custom block.

    :param data_dtype: The data-type (inferred from a column) as prescribed from ``api.dtype``

    :returns: The appropriate function that will pre-process (clean) data of specified dtype.
    """ # noqa
    if data_dtype in custom_cleaning_functions:
        clean_func = eval(custom_cleaning_functions[data_dtype])

    elif data_dtype in (dtype.date, dtype.datetime):
        clean_func = _standardize_datetime

    elif data_dtype in (dtype.float, dtype.num_tsarray):
        clean_func = _clean_float

    elif data_dtype in (dtype.integer):
        clean_func = _clean_int

    elif data_dtype in (dtype.num_array):
        clean_func = _standardize_num_array

    elif data_dtype in (dtype.cat_array):
        clean_func = _standardize_cat_array

    elif data_dtype in (dtype.tags):
        clean_func = _tags_to_tuples

    elif data_dtype in (dtype.quantity):
        clean_func = _clean_quantity

    elif data_dtype in (
        dtype.short_text,
        dtype.rich_text,
        dtype.categorical,
        dtype.binary,
        dtype.audio,
        dtype.image,
        dtype.video,
        dtype.cat_tsarray
    ):
        clean_func = _clean_text

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
    if element is None or pd.isna(element):
        return 0.0  # correct? TODO: Remove if the TS encoder can handle `None`
    try:
        date = parse_dt(str(element))
    except Exception:
        try:
            date = datetime.datetime.utcfromtimestamp(element)
        except Exception:
            return 0.0

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


def _standardize_num_array(element: object) -> Optional[Union[List[float], float]]:
    """
    Given an array of numbers in the form ``[1, 2, 3, 4]``, converts into a numerical sequence.

    :param element: An array-like element in a sequence
    :returns: standardized array OR scalar number IF edge case

    Ex of edge case:
    >> element = [1]
    >> _standardize_num_array(element)
    >> 1
    """
    try:
        element = str(element)
        element = element.rstrip("]").lstrip("[")
        element = element.rstrip(" ").lstrip(" ")
        element = element.replace(", ", " ").replace(",", " ")
        # Handles cases where arrays are numbers
        if " " not in element:
            element = _clean_float(element)
        else:
            element = [float(x) for x in element.split(" ")]
    except Exception:
        pass

    return element


def _standardize_cat_array(element: List) -> Optional[List[str]]:
    """
    Given an array, replace non-string with string casted (or, failing that, None) tokens. None values are kept.

    :param element: An array-like element in a sequence
    :returns: standardized array OR label IF edge case

    Ex of edge case:
    >> element = ['a']
    >> _standardize_cat_array(element)
    >> 'a'
    """
    if len(element) == 1:
        return element[0]
    else:
        new_element = []
        for sub_elt in element:
            try:
                new_sub_elt = str(sub_elt) if sub_elt else None
                new_element.append(new_sub_elt)
            except TypeError:
                new_element.append(None)
        element = new_element

    return element


# ------------------------- #
# Integers/Floats/Quantities
# ------------------------- #

def _clean_float(element: object) -> Optional[float]:
    """
    Given an element, converts it into float numeric format. If element is NaN, or inf, then returns None.
    """
    try:
        cleaned_float = text.clean_float(element)
        if is_nan_numeric(cleaned_float):
            return None
        return cleaned_float
    except Exception:
        return None


def _clean_int(element: object) -> Optional[int]:
    """
    Given an element, converts it into integer numeric format. If element is NaN, or inf, then returns None.
    """
    element = _clean_float(element)
    if element is not None:
        element = int(element)
    return element


def _clean_quantity(element: object) -> Optional[float]:
    """
    Given a quantity, clean and convert it into float numeric format. If element is NaN, or inf, then returns None.
    """
    no_symbols = re.sub("[^0-9.,]", "", str(element)).replace(",", ".")
    no_symbols = '0' if no_symbols == '' else no_symbols
    element = float(no_symbols)
    return _clean_float(element)


# ------------------------- #
# Text
# ------------------------- #
def _clean_text(element: object) -> Union[str, None]:
    if isinstance(element, str):
        return element
    elif element is None or element != element:
        return None
    else:
        return str(element)


# ------------------------- #
# Other helpers
# ------------------------- #
def _rm_rows_w_empty_targets(df: pd.DataFrame, target: str) -> pd.DataFrame:
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


def _remove_columns(data: pd.DataFrame, identifiers: Dict[str, object], target: str,
                    mode: str, timeseries_settings: TimeseriesSettings, anomaly_detection: bool,
                    dtype_dict: Dict[str, dtype]) -> pd.DataFrame:
    """
    Drop columns we don't want to use in order to train or predict

    :param data: The raw data
    :param dtype_dict: Type information for each column
    :param identifiers: A dict containing all identifier typed columns
    :param target: The target columns
    :param mode: Can be "predict" or "train"
    :param timeseries_settings: Timeseries related settings, only relevant for timeseries predictors, otherwise can be the default object
    :param anomaly_detection: Are we detecting anomalies with this predictor?

    :returns: A (new) dataframe without the dropped columns
    """ # noqa
    data = deepcopy(data)
    to_drop = [*[x for x in identifiers.keys() if x != target],
               *[x for x in data.columns if x in dtype_dict and dtype_dict[x] == dtype.invalid]]

    exceptions = ["__mdb_forecast_offset"]
    if timeseries_settings.group_by is not None:
        exceptions += timeseries_settings.group_by

    to_drop = [x for x in to_drop if x in data.columns and x not in exceptions]

    if to_drop:
        log.info(f'Dropping features: {to_drop}')
        data = data.drop(columns=to_drop)

    if mode == "train":
        data = _rm_rows_w_empty_targets(data, target)
    if mode == "predict":
        if (
            target in data.columns
            and (not timeseries_settings.is_timeseries or not timeseries_settings.use_previous_target)
            and not anomaly_detection
        ):
            data = data.drop(columns=[target])

    # Drop extra columns
    for name in list(data.columns):
        if name not in dtype_dict and name not in exceptions:
            data = data.drop(columns=[name])
    return data


def _get_columns_to_clean(data: pd.DataFrame, dtype_dict: Dict[str, dtype], mode: str, target: str) -> List[str]:
    """
    :param data: The raw data
    :param dtype_dict: Type information for each column
    :param target: The target columns
    :param mode: Can be "predict" or "train"

    :returns: A list of columns that we want to clean
    """ # noqa

    cleanable_columns = []
    for name, _ in dtype_dict.items():
        if mode == "predict":
            if name == target:
                continue
        if name in data.columns:
            cleanable_columns.append(name)
    return cleanable_columns


def clean_timeseries(df: pd.DataFrame, tss: TimeseriesSettings) -> pd.DataFrame:
    """
    All timeseries-specific cleaning logic goes here. Currently:

    1) Any row with `nan`-valued order-by measurements is dropped.

    :param df: data.
    :param tss: timeseries settings
    :return: cleaned data.
    """
    invalid_rows = []

    for idx, row in df.iterrows():
        if pd.isna(row[tss.order_by]):
            invalid_rows.append(idx)

    df = df.drop(invalid_rows)

    return df
