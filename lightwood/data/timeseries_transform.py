from lightwood.api.types import TimeseriesSettings
import pandas as pd


def transform_timeseries(data: pd.DataFrame, timeseries_settings: TimeseriesSettings) -> pd.DataFrame:
    return data