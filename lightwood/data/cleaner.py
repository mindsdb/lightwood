import re
from tests.unit_tests import data
from typing import Dict, List
from lightwood.api.dtype import dtype
from lightwood.api.types import Feature, JsonML, Output
from lightwood.helpers.log import log
from dateutil.parser import parse as parse_dt
import datetime
from mindsdb_datasources import DataSource
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

    return element


def cleaner(data: pd.DataFrame, dtype_dict: Dict[str, str], pct_invalid: int, ignore_features: List[str], identifiers: Dict[str, str]) -> pd.DataFrame:
    # Drop columns we don't want to use
    to_drop = [*ignore_features, *list(identifiers.keys())]
    data = data.drop(columns=to_drop)

    # Drop extra columns
    for name in list(data.columns):
        if name not in dtype_dict:
            data = data.drop(columns=[name])

    # Standardize content
    for name, data_dtype in dtype_dict.items():
        if name in to_drop:
            continue
        if name not in data.columns:
            data[name] = [None] * len(data)
            continue

        new_data = []
        for element in data[name]:
            try:
                new_data.append(_clean_value(element, data_dtype))
            except Exception as e:
                new_data.append(None)
                log.warning(f'Unable to parse elemnt: {element} or type {data_dtype} from column {name}. Excetpion: {e}')

        pct_invalid = 100 * (len(new_data) - len([x for x in new_data if x is not None])) / len(new_data)

        if pct_invalid > pct_invalid:
            err = f'Too many ({pct_invalid}%) invalid values in column {name} of type {data_dtype}'
            log.error(err)
            raise Exception(err)

        data[name] = new_data

    return data
