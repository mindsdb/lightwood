import re
from copy import deepcopy

import numpy as np
import pandas as pd

# For time-series
import datetime
from dateutil.parser import parse as parse_dt

from lightwood.api.dtype import dtype
from lightwood.helpers import text
from lightwood.helpers.log import log
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.numeric import can_be_nan_numeric

# Import NLTK for stopwords
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

from typing import Dict, List, Optional, Tuple, Callable, Union

# Borrow functions from Lightwood's cleaner
from lightwood.data.cleaner import (
    _remove_columns,
    _get_columns_to_clean,
    get_cleaning_func,
)

# Use for standardizing NaNs
VALUES_FOR_NAN_AND_NONE_IN_PANDAS = [np.nan, "nan", "NaN", "Nan", "None"]


def cleaner(
    data: pd.DataFrame,
    dtype_dict: Dict[str, str],
    identifiers: Dict[str, str],
    target: str,
    mode: str,
    timeseries_settings: TimeseriesSettings,
    anomaly_detection: bool,
    custom_cleaning_functions: Dict[str, str] = {},
) -> pd.DataFrame:
    """
    The cleaner is a function which takes in the raw data, plus additional information about it's types and about the problem. Based on this it generates a "clean" representation of the data, where each column has an ideal standardized type and all malformed or otherwise missing or invalid elements are turned into ``None``

    :param data: The raw data
    :param dtype_dict: Type information for each column
    :param identifiers: A dict containing all identifier typed columns
    :param target: The target columns
    :param mode: Can be "predict" or "train"
    :param timeseries_settings: Timeseries related settings, only relevant for timeseries predictors, otherwise can be the default object
    :param anomaly_detection: Are we detecting anomalies with this predictor?

    :returns: The cleaned data
    """  # noqa

    data = _remove_columns(
        data,
        identifiers,
        target,
        mode,
        timeseries_settings,
        anomaly_detection,
        dtype_dict,
    )

    for col in _get_columns_to_clean(data, dtype_dict, mode, target):

        log.info("Cleaning column =" + str(col))
        # Get and apply a cleaning function for each data type
        # If you want to customize the cleaner, it's likely you can to modify ``get_cleaning_func``
        data[col] = data[col].apply(
            get_cleaning_func(dtype_dict[col], custom_cleaning_functions)
        )

        # ------------------------ #
        # INTRODUCE YOUR CUSTOM BLOCK

        # If column data type is a text type, remove stop-words
        if dtype_dict[col] in (dtype.rich_text, dtype.short_text):
            data[col] = data[col].apply(
                lambda x: " ".join(
                    [word for word in x.split() if word not in stop_words]
                )
            )

        # Enforce numerical columns as non-negative
        if dtype_dict[col] in (dtype.integer, dtype.float):
            log.info("Converted " + str(col) + " into strictly non-negative")
            data[col] = data[col].apply(lambda x: x if x > 0 else 0.0)

        # ------------------------ #
        data[col] = data[col].replace(
            to_replace=VALUES_FOR_NAN_AND_NONE_IN_PANDAS, value=None
        )

    return data
