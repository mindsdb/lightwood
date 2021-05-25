from mindsdb_datasources import DataSource
import pandas as pd


def cleaner(data: DataSource) -> pd.DataFrame:
    return data.df
