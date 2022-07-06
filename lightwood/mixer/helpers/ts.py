"""
Time series utility methods for usage within mixers.
"""
from typing import Dict
from copy import deepcopy

import pandas as pd

from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_group_matches, get_ts_groups


def _transform_target(ts_analysis: Dict[str, Dict], target_df: pd.DataFrame, group: tuple):
    target_df = deepcopy(target_df)  # needed because methods like get_level_values actually have undisclosed side effects  # noqa
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(target_df.index, pd.MultiIndex) and len(target_df.index.levels) > 1:
        return transformer.transform(target_df.droplevel(0).to_period())
    elif isinstance(target_df.index, pd.MultiIndex):
        temp_s = pd.Series(data=target_df.values)
        temp_s.index = target_df.index.levels[0]  # warning: do not use get_level_values as it removes inferred freq
        return transformer.transform(temp_s.to_period())
    else:
        return transformer.transform(target_df.to_period())


def _inverse_transform_target(ts_analysis: Dict[str, Dict], predictions: pd.DataFrame, group: tuple):
    predictions = deepcopy(predictions)  # needed because methods like get_level_values actually have undisclosed side effects  # noqa
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(predictions.index, pd.MultiIndex) and len(predictions.index.levels) > 1:
        return transformer.inverse_transform(predictions.droplevel(0).to_period())
    elif isinstance(predictions.index, pd.MultiIndex):
        temp_s = pd.Series(data=predictions)
        temp_s.index = predictions.index.levels[0]  # warning: do not use get_level_values as it removes inferred freq
        return transformer.transform(temp_s.to_period())
    else:
        return transformer.inverse_transform(predictions.to_period())


def _apply_stl_on_training(
        train_data: EncodedDs,
        dev_data: EncodedDs,
        target: str,
        tss: TimeseriesSettings,
        ts_analysis: Dict[str, Dict],
):
    """
    The purpose of this function is a side-effect:
        applying STL blocks in the target column of the dataframes embedded
        in the passed EncodedDs instances.
        
    :param train_data: EncodedDs object for the training dataset 
    :param dev_data: EncodedDs object for the dev/validation dataset
    :return: 
    """  # noqa
    gby = tss.group_by if tss.group_by else []
    tr_midx = pd.MultiIndex.from_frame(train_data.data_frame.reset_index()[[*gby, 'index']])
    dev_midx = pd.MultiIndex.from_frame(dev_data.data_frame.reset_index()[[*gby, 'index']])
    train_data.data_frame.index = tr_midx
    dev_data.data_frame.index = dev_midx

    for group in ts_analysis['group_combinations']:
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            train_idxs, train_subset = get_group_matches(train_data.data_frame, group, gby)
            dev_idxs, dev_subset = get_group_matches(dev_data.data_frame, group, gby)

            train_data.data_frame[target].loc[train_idxs] = _transform_target(ts_analysis,
                                                                              train_subset[target], group).values

            dev_data.data_frame[target].loc[dev_idxs] = _transform_target(ts_analysis,
                                                                          dev_subset[target], group).values

            # shift all timestep cols here by respective offset
            for timestep in range(1, tss.horizon):
                train_data.data_frame[f'{target}_timestep_{timestep}'].loc[train_idxs] = \
                    train_data.data_frame[target].loc[train_idxs].shift(-timestep)

                dev_data.data_frame[f'{target}_timestep_{timestep}'].loc[dev_idxs] = \
                    dev_data.data_frame[target].loc[dev_idxs].shift(-timestep)


def _stl_transform(
        ydf: pd.DataFrame,
        ds: EncodedDs,
        target: str,
        tss: TimeseriesSettings,
        ts_analysis: Dict[str, Dict]
) -> pd.DataFrame:
    """
   The purpose of this function is a side-effect:
       applying STL blocks in the target column of the dataframes embedded
       in the passed EncodedDs instances.

    :param ydf: 
    :param ds: 
    :param target: 
    :param tss: 
    :param ts_analysis: 
    :return: 
    """  # noqa
    gby = tss.group_by if tss.group_by else []
    midx = pd.MultiIndex.from_frame(ds.data_frame.reset_index()[[*gby, 'index']])
    midx.levels[0].freq = ds.data_frame['__mdb_inferred_freq'].iloc[0]
    ds.data_frame.index = midx
    ydf.index = midx
    groups = get_ts_groups(ds.data_frame, tss)
    for group in groups:
        group = group if group in ts_analysis['group_combinations'] else '__default'
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            idxs, subset = get_group_matches(ds.data_frame, group, gby)
            ds.data_frame[target].loc[idxs] = _transform_target(
                ts_analysis, subset[target], group).values
    return ds.data_frame  # TODO: check that the side effects actually worked outside the fn scope and avoid returning?


def _stl_inverse_transform(
        ydf: pd.DataFrame,
        ds: EncodedDs,
        tss: TimeseriesSettings,
        ts_analysis: Dict[str, Dict]
) -> None:
    """
    :param ydf: 
    :param ds: 
    :param target: 
    :param tss: 
    :param ts_analysis: 
    :return: 
    """  # noqa
    groups = get_ts_groups(ds.data_frame, tss)
    gby = tss.group_by if tss.group_by else []
    for group in groups:
        group = group if group in ts_analysis['group_combinations'] else '__default'
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            idxs, subset = get_group_matches(ds.data_frame, group, gby)
            for timestep in range(tss.horizon):
                ydf[f'prediction_{timestep}'].loc[idxs] = _inverse_transform_target(ts_analysis,
                                                                                    ydf[f'prediction_{timestep}'].loc[
                                                                                        idxs],
                                                                                    group
                                                                                    ).values
    return ydf.reset_index(drop=True)
