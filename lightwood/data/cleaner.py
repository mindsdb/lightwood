from typing import Dict
from lightwood.api.dtype import dtype
from lightwood.api.types import Feature, LightwoodConfig, Output
from lightwood.helpers.log import log
from dateutil.parser import parse as parse_dt
from mindsdb_datasources import DataSource
import pandas as pd


def clean_value(element: object, data_dtype: str):
    if data_dtype in (dtype.date, dtype.datetime):
        element = parse_dt(element)
    return element


def cleaner(data: DataSource, dtype_dict: Dict[str, str], pct_invalid: int) -> pd.DataFrame:
    for name, data_dtype in dtype_dict:
        new_data = []
        for element in data.df[name]:
            try:
                new_data.append(clean_value(element, data_dtype))
            except Exception as e:
                new_data.append(None)
                log.warning(f'Unable to parse elemnt: {element} or type {data_dtype} from column {name}. Excetpion: {e}')

        pct_invalid = 100 * (len(new_data) - len([x for x in new_data if x is not None])) / len(new_data)
        
        if pct_invalid > pct_invalid:
            err = f'Too many ({pct_invalid}%) invalid values in column {name} of type {data_dtype}'
            log.error(err)
            raise Exception(err)

    return data.df
