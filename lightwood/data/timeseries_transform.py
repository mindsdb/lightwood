from typing import Dict
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from lightwood.helpers.parallelism import get_nr_procs
from lightwood.helpers.ts import get_ts_groups, get_delta, get_group_matches

from type_infer.dtype import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.log import log


def transform_timeseries(
        data: pd.DataFrame, dtype_dict: Dict[str, str], ts_analysis: dict,
        timeseries_settings: TimeseriesSettings, target: str, mode: str) -> pd.DataFrame:
    """
    Block that transforms the dataframe of a time series task to a convenient format for use in posterior phases like model training.
    
    The main transformations performed by this block are:
      - Type casting (e.g. to numerical for `order_by` column).
      - Windowing functions for historical context based on `TimeseriesSettings.window` parameter.
      - Explicitly add target columns according to the `TimeseriesSettings.horizon` parameter.
      - Flag all rows that are "predictable" based on all `TimeseriesSettings`.
      - Plus, handle all logic for the streaming use case (where forecasts are only emitted for the last observed data point).
    
    :param data: Dataframe with data to transform.
    :param dtype_dict: Dictionary with the types of each column.
    :param ts_analysis: dictionary with various insights into each series passed as training input.
    :param timeseries_settings: A `TimeseriesSettings` object.
    :param target: The name of the target column to forecast.
    :param mode: Either "train" or "predict", depending on what phase is calling this procedure.
    
    :return: A dataframe with all the transformations applied.
    """  # noqa

    tss = timeseries_settings
    gb_arr = tss.group_by if tss.group_by is not None else []
    oby = tss.order_by
    window = tss.window

    if tss.use_previous_target and target not in data.columns:
        raise Exception(f"Cannot transform. Missing historical values for target column {target} (`use_previous_target` is set to True).")  # noqa

    for hcol in tss.historical_columns:
        if hcol not in data.columns or data[hcol].isna().any():
            raise Exception(f"Cannot transform. Missing values in historical column {hcol}.")

    # infer frequency with get_delta
    oby_col = tss.order_by
    groups = get_ts_groups(data, tss)

    # initial stable sort and per-partition deduplication
    data = data.sort_values(by=oby_col, kind='mergesort')
    data = data.drop_duplicates(subset=[oby_col, *gb_arr], keep='first')

    if not ts_analysis:
        _, periods, freqs = get_delta(data, dtype_dict, groups, target, tss)
    else:
        periods = ts_analysis['periods']
        freqs = ts_analysis['sample_freqs']

    # pass seconds to timestamps according to each group's inferred freq, and force this freq on index
    subsets = []
    for group in groups:
        if (tss.group_by and group != '__default') or not tss.group_by:
            idxs, subset = get_group_matches(data, group, tss.group_by)
            if subset.shape[0] > 0:
                if periods.get(group, periods['__default']) == 0 and subset.shape[0] > 1:
                    raise Exception(
                        f"Partition is not valid, faulty group {group}. Please make sure you group by a set of columns that ensures unique measurements for each grouping through time.")  # noqa

                index = pd.to_datetime(subset[oby_col], unit='s')
                subset.index = pd.date_range(start=index.iloc[0],
                                             freq=freqs.get(group, freqs['__default']),
                                             periods=len(subset))
                subset['__mdb_inferred_freq'] = subset.index.freq   # sets constant column because pd.concat forgets freq (see: https://github.com/pandas-dev/pandas/issues/3232)  # noqa
                subsets.append(subset)
    original_df = pd.concat(subsets).sort_values(by='__mdb_original_index')

    if '__mdb_forecast_offset' in original_df.columns:
        """ This special column can be either None or an integer. If this column is passed, then the TS transformation will react to the values within:

        * If all rows = `None`, proceed as usual. This ends up generating one HORIZON-length forecast for each row in the DF.
        * If all rows have the same value `N <= 0`, then cutoff the dataframe latest `-N` rows after TS shaping and prime the DF (with `__make_predictions` column) so that a forecast is generated only for the last row (thus more efficient). This enables `WHERE T = LATEST - K` with `0 <= K < WINDOW` syntax upstream in MindsDB.
        * If all rows have the same value `N = 1`, then activate streaming inference mode where a single forecast will be emitted for the timestamp inferred by the `_ts_infer_next_row` method. This enables the (already supported) `WHERE T > LATEST` syntax.
        """  # noqa
        index = original_df[~original_df['__mdb_forecast_offset'].isin([None])]  # trigger if col is constant & != None
        offset_available = index.shape[0] == len(original_df) and \
            original_df['__mdb_forecast_offset'].unique().tolist() != [None]
        if offset_available:
            offset = min(int(original_df['__mdb_forecast_offset'].unique()[0]), 1)
        else:
            offset = 0
        infer_mode = offset_available and offset == 1
    else:
        offset_available = False
        offset = 0
        infer_mode = False

    original_index_list = []
    idx = 0
    for row in original_df.itertuples():
        if _make_pred(row) or infer_mode:
            original_df.at[row.Index, '__make_predictions'] = True
            original_index_list.append(idx)
            idx += 1
        else:
            original_df.at[row.Index, '__make_predictions'] = False
            original_index_list.append(None)

    original_df['original_index'] = original_index_list

    secondary_type_dict = {}
    if dtype_dict[oby] in (dtype.date, dtype.integer, dtype.float):
        secondary_type_dict[oby] = dtype_dict[oby]

    original_df[f'__mdb_original_{oby}'] = original_df[oby]
    group_lengths = []
    if len(gb_arr) > 0:
        df_arr = []
        for _, df in original_df.groupby(gb_arr):
            df_arr.append(df.sort_values(by=oby))
            group_lengths.append(len(df))
    else:
        df_arr = [original_df.sort_values(by=oby)]
        group_lengths.append(len(original_df))

    n_groups = len(df_arr)
    for i, subdf in enumerate(df_arr):
        if '__mdb_forecast_offset' in subdf.columns and mode == 'predict':
            if infer_mode:
                df_arr[i] = _ts_infer_next_row(subdf, oby)
                make_preds = [False for _ in range(max(0, len(df_arr[i]) - 1))] + [True]
            elif offset_available:
                # truncate to forecast up until some len(df) + offset (which is <= 0)
                new_index = df_arr[i].index[:len(df_arr[i].index) + offset]
                make_preds = [False for _ in range(max(0, len(new_index) - 1))] + [True]
                df_arr[i] = df_arr[i].loc[new_index]
            else:
                make_preds = [True for _ in range(len(df_arr[i]))]
            df_arr[i]['__make_predictions'] = make_preds

    if len(original_df) > 500:
        # @TODO: restore possibility to override this with args
        nr_procs = get_nr_procs(original_df)
        log.info(f'Using {nr_procs} processes to reshape.')
        pool = mp.Pool(processes=nr_procs)
        # Make type `object` so that dataframe cells can be python lists
        df_arr = pool.map(partial(_ts_to_obj, historical_columns=[oby] + tss.historical_columns), df_arr)
        df_arr = pool.map(
            partial(
                _ts_add_previous_rows, order_cols=[oby] + tss.historical_columns, window=window),
            df_arr)
        df_arr = pool.map(partial(_ts_add_future_target, target=target, horizon=tss.horizon,
                                  data_dtype=tss.target_type, mode=mode),
                          df_arr)

        if tss.use_previous_target:
            df_arr = pool.map(
                partial(_ts_add_previous_target, target=target, window=tss.window),
                df_arr)
        pool.close()
        pool.join()
    else:
        for i in range(n_groups):
            df_arr[i] = _ts_to_obj(df_arr[i], historical_columns=[oby] + tss.historical_columns)
            df_arr[i] = _ts_add_previous_rows(df_arr[i],
                                              order_cols=[oby] + tss.historical_columns, window=window)
            df_arr[i] = _ts_add_future_target(df_arr[i], target=target, horizon=tss.horizon,
                                              data_dtype=tss.target_type, mode=mode)
            if tss.use_previous_target:
                df_arr[i] = _ts_add_previous_target(df_arr[i], target=target, window=tss.window)

    combined_df = pd.concat(df_arr)

    if '__mdb_forecast_offset' in combined_df.columns:
        combined_df = pd.DataFrame(combined_df[combined_df['__make_predictions']])  # filters by True only
        del combined_df['__make_predictions']

    if not infer_mode and any([i < tss.window for i in group_lengths]):
        if tss.allow_incomplete_history:
            log.warning("Forecasting with incomplete historical context, predictions might be subpar")
        else:
            raise Exception(f'Not enough historical context to make a timeseries prediction (`allow_incomplete_history` is set to False). Please provide a number of rows greater or equal to the window size - currently (number_rows, window_size) = ({min(group_lengths)}, {tss.window}). If you can\'t get enough rows, consider lowering your window size. If you want to force timeseries predictions lacking historical context please set the `allow_incomplete_history` timeseries setting to `True`, but this might lead to subpar predictions depending on the mixer.') # noqa

    df_gb_map = None
    if n_groups > 1:
        df_gb_list = list(combined_df.groupby(tss.group_by))
        df_gb_map = {}
        for gb, df in df_gb_list:
            df_gb_map['_' + '_'.join(str(gb))] = df

    timeseries_row_mapping = {}
    idx = 0

    if df_gb_map is None:
        for i in range(len(combined_df)):
            row = combined_df.iloc[i]
            if not infer_mode:
                timeseries_row_mapping[idx] = int(
                    row['original_index']) if row['original_index'] is not None and not np.isnan(
                    row['original_index']) else None
            else:
                timeseries_row_mapping[idx] = idx
            idx += 1
    else:
        for gb in df_gb_map:
            for i in range(len(df_gb_map[gb])):
                row = df_gb_map[gb].iloc[i]
                if not infer_mode:
                    timeseries_row_mapping[idx] = int(
                        row['original_index']) if row['original_index'] is not None and not np.isnan(
                        row['original_index']) else None
                else:
                    timeseries_row_mapping[idx] = idx

                idx += 1

    del combined_df['original_index']

    return combined_df


def _ts_infer_next_row(df: pd.DataFrame, ob: str) -> pd.DataFrame:
    """
    Adds an inferred next row for streaming mode purposes.

    :param df: dataframe from which next row is inferred.
    :param ob: `order_by` column.

    :return: Modified `df` with the inferred row appended to it.
    """
    original_index = df.index.copy()
    start = original_index.min()
    new_index = pd.date_range(start=start, periods=len(original_index) + 1, freq=df['__mdb_inferred_freq'].iloc[0])
    last_row = df.iloc[[-1]].copy()
    last_row['__mdb_ts_inferred'] = True

    if df.shape[0] > 1:
        butlast_row = df.iloc[[-2]]
        delta = (last_row[ob].values - butlast_row[ob].values).flatten()[0]
    else:
        delta = 1

    last_row[ob] += delta
    new_df = df.append(last_row)
    new_df.index = pd.DatetimeIndex(new_index)
    return new_df


def _make_pred(row) -> bool:
    """
    Indicates whether a prediction should be made for `row` or not.
    """
    return not hasattr(row, '__mdb_forecast_offset') or row.make_predictions


def _ts_to_obj(df: pd.DataFrame, historical_columns: list) -> pd.DataFrame:
    """
    Casts all historical columns in a dataframe to `object` type.

    :param df: Input dataframe
    :param historical_columns: Historical columns to type cast

    :return: Dataframe with `object`-typed historical columns
    """
    for hist_col in historical_columns:
        df.loc[:, hist_col] = df[hist_col].astype(object)
    return df


def _ts_add_previous_rows(df: pd.DataFrame, order_cols: list, window: int) -> pd.DataFrame:
    """
    Adds previous rows (as determined by `TimeseriesSettings.window`) into the cells of the `order_by` column.

    :param df: Input dataframe.
    :param order_cols: `order_by` column and other columns flagged as `historical`.
    :param window: value of `TimeseriesSettings.window` parameter.
    
    :return: Dataframe with all `order_cols` modified so that their values are now arrays of historical context.
    """  # noqa
    for order_col in order_cols:
        new_vals = np.zeros((len(df), window))
        for i in range(window, 0, -1):
            new_vals[:, i - 1] = df[order_col].shift(window - i).values

        new_vals = np.nan_to_num(new_vals, nan=0.0)
        for i in range(len(df)):
            df.at[df.index[i], order_col] = new_vals[i, :]

    return df


def _ts_add_previous_target(df: pd.DataFrame, target: str, window: int) -> pd.DataFrame:
    """
    Adds previous rows (as determined by `TimeseriesSettings.window`) into the cells of the target column.

    :param df: Input dataframe.
    :param target: target column name.
    :param window: value of `TimeseriesSettings.window` parameter.

    :return: Dataframe with new `__mdb_ts_previous_{target}` column that contains historical target context.
    """  # noqa
    if target not in df:
        return df
    previous_target_values = list(df[target])
    del previous_target_values[-1]
    previous_target_values = [None] + previous_target_values

    previous_target_values_arr = []
    for i in range(len(previous_target_values)):
        prev_vals = previous_target_values[max(i - window, 0):i + 1]
        arr = [None] * (window - len(prev_vals))
        arr.extend(prev_vals)
        previous_target_values_arr.append(arr)

    df[f'__mdb_ts_previous_{target}'] = previous_target_values_arr
    return df


def _ts_add_future_target(df, target, horizon, data_dtype, mode):
    """
    Adds as many columns to the input dataframe as the forecasting horizon asks for (as determined by `TimeseriesSettings.horizon`).

    :param df: Input dataframe.
    :param target: target column name.
    :param horizon: value of `TimeseriesSettings.horizon` parameter.
    :param data_dtype: dictionary with types of all input columns
    :param mode: either "train" or "predict". `Train` will drop rows with incomplet target info. `Predict` has no effect, for now.

    :return: Dataframe with new `{target}_timestep_{i}'` columns that contains target labels at timestep `i` of a total `TimeseriesSettings.horizon`.
    """  # noqa
    if target not in df:
        return df
    if data_dtype in (dtype.integer, dtype.float, dtype.num_array, dtype.num_tsarray):
        df[target] = df[target].astype(float)

    for timestep_index in range(1, horizon):
        next_target_value_arr = list(df[target])
        for del_index in range(0, min(timestep_index, len(next_target_value_arr))):
            del next_target_value_arr[0]
            next_target_value_arr.append(None)
        col_name = f'{target}_timestep_{timestep_index}'
        df[col_name] = next_target_value_arr
        df[col_name] = df[col_name].fillna(value=np.nan)

    return df
