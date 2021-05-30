from lightwood.api.dtype import dtype
from lightwood.api.types import LightwoodConfig
from lightwood.helpers.log import log
from dateutil.parser import parse as parse_dt
from mindsdb_datasources import DataSource
import pandas as pd


def clean_value(element: object, data_dtype: str):
    if data_dtype in (dtype.date, dtype.datetime):
        element = parse_dt(element)
    return element

def cleaner(data: DataSource, lightwood_config: LightwoodConfig) -> pd.DataFrame:
    for col_data in [*lightwood_config.features.values(), lightwood_config.output]:
        new_data = []
        for element in data.df[col_data.name]:
            try:
                new_data.append(clean_value(element, col_data.dtype))
            except Exception as e:
                new_data.append(None)
                log.warning(f'Unable to parse elemnt: {element} or type {col_data.dtype} from column {col_data.name}. Excetpion: {e}')
        
        pct_invalid = 100 - len([x for x in new_data if x is not None]) / len(new_data)
        if pct_invalid > lightwood_config.problem_definition.pct_invalid:
            err = 'Too many ({pct_invalid}%) invalid values in column {col_data.name} of type {col_data.dtype}'
            log.error(err)
            raise Exception(err)

    return data.df
