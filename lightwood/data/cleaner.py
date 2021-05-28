from lightwood.api.dtype import dtype
from lightwood.api.types import LightwoodConfig
from dateutil.parser import parse as parse_dt
from mindsdb_datasources import DataSource
import pandas as pd


def cleaner(data: DataSource, lightwood_config: LightwoodConfig) -> pd.DataFrame:
    for col_data in [*lightwood_config.features.values(), lightwood_config.output]:
        if col_data.data_dtype in (dtype.date, dtype.datetime):
            data.df[col_data.name] = [parse_dt(x) for x in data.df[col_data.name]]

    return data.df
