from copy import deepcopy
from lightwood.api.types import TimeseriesSettings
import re
from typing import Dict, List
from lightwood.api.dtype import dtype
from lightwood.helpers.log import log
from dateutil.parser import parse as parse_dt
import datetime
from lightwood.helpers.text import clean_float
import pandas as pd


def _to_datetime(element):
    try:
        date = parse_dt(element)
    except Exception:
        try:
            date = datetime.datetime.utcfromtimestamp(element)
        except Exception:
            return None

    return date


def _standardize_date(element):
    date = _to_datetime(element)
    if date is None:
        return None
    return date.timestamp()


def _standardize_datetime(element):
    date = _to_datetime(element)
    if date is None:
        return None
    return date.timestamp()


def _tags_to_tuples(tags_str):
    try:
        return tuple([x.strip() for x in tags_str.split(',')])
    except Exception:
        return tuple()


def _standardize_array(element):
    try:
        element = str(element)
        element = element.rstrip(']').lstrip('[')
        element = element.rstrip(' ').lstrip(' ')
        return element.replace(', ', ' ').replace(',', ' ')
    except Exception:
        return element


def _clean_float_or_none(element):
    try:
        return clean_float(element)
    except Exception:
        return None


def _clean_value(element: object, data_dtype: str):
    if data_dtype in (dtype.date):
        element = _standardize_date(element)

    if data_dtype in (dtype.datetime):
        element = _standardize_datetime(element)

    if data_dtype in (dtype.float):
        element = _clean_float_or_none(element)
    if data_dtype in (dtype.integer):
        element = int(_clean_float_or_none(element))

    if data_dtype in (dtype.array):
        element = _standardize_array(element)

    if data_dtype in (dtype.tags):
        element = _tags_to_tuples(element)

    if data_dtype in (dtype.quantity):
        element = float(re.sub("[^0-9.,]", '', element).replace(',', '.'))

    if data_dtype in (dtype.short_text, dtype.rich_text, dtype.categorical, dtype.binary):
        element = str(element)

    return element


def clean_empty_targets(df: pd.DataFrame, target: str) -> pd.DataFrame:
    len_before = len(df)
    df = df.dropna(subset=[target])
    len_after = len(df)
    nr_removed = len_before - len_after
    if nr_removed != 0:
        log.warning(
            f'Removed {nr_removed} rows due to the target value missing. Training with rows without a target value makes no sense, please avoid this!') # noqa

    return df


def cleaner(
        data: pd.DataFrame, dtype_dict: Dict[str, str],
        pct_invalid: float, ignore_features: List[str],
        identifiers: Dict[str, str],
        target: str, mode: str, timeseries_settings: TimeseriesSettings, anomaly_detection: bool) -> pd.DataFrame:
    # Drop columns we don't want to use
    data = deepcopy(data)
    to_drop = [*ignore_features, *list(identifiers.keys())]
    exceptions = ['__mdb_make_predictions']
    for col in to_drop:
        try:
            data = data.drop(columns=[col])
        except Exception:
            pass

    if mode == 'train':
        data = clean_empty_targets(data, target)
    if mode == 'predict':
        if target in data.columns and not timeseries_settings.use_previous_target and not anomaly_detection:
            data = data.drop(columns=[target])

    # Drop extra columns
    for name in list(data.columns):
        if name not in dtype_dict and name not in exceptions:
            data = data.drop(columns=[name])

    # Standardize content
    for name, data_dtype in dtype_dict.items():
        if mode == 'predict':
            if name == target:
                continue
        if name in to_drop:
            continue
        if name not in data.columns:
            if '__mdb_ts_previous' not in name:
                data[name] = [None] * len(data)
            continue

        new_data = []
        for element in data[name]:
            try:
                new_data.append(_clean_value(element, data_dtype))
            except Exception as e:
                new_data.append(None)
                log.warning(
                    f'Unable to parse elemnt: {element} or type {data_dtype} from column {name}. Excetpion: {e}')

        pct_invalid = 100 * (len(new_data) - len([x for x in new_data if x is not None])) / len(new_data)

        if pct_invalid > pct_invalid:
            err = f'Too many ({pct_invalid}%) invalid values in column {name} of type {data_dtype}'
            log.error(err)
            raise Exception(err)

        data[name] = new_data

    return data
