"""
Time series utility methods for usage within mixers.
"""
from typing import Dict

import pandas as pd

from lightwood.data.encoded_ds import EncodedDs
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_group_matches, get_ts_groups


def _transform_target(ts_analysis: Dict[str, Dict], target_df: pd.DataFrame, group: tuple):
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(target_df.index, pd.MultiIndex):
        return transformer.transform(target_df.droplevel(0).to_period())
    else:
        return transformer.transform(target_df.to_period())


def _inverse_transform_target(ts_analysis: Dict[str, Dict], predictions: pd.DataFrame, group: tuple):
    transformer = ts_analysis['stl_transforms'][group]['transformer']
    if isinstance(predictions.index, pd.MultiIndex):
        return transformer.inverse_transform(predictions.droplevel(0).to_period())
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
    tr_midx = pd.MultiIndex.from_frame(train_data.data_frame.reset_index()[[*tss.group_by, 'index']])
    dev_midx = pd.MultiIndex.from_frame(dev_data.data_frame.reset_index()[[*tss.group_by, 'index']])
    train_data.data_frame.index = tr_midx
    dev_data.data_frame.index = dev_midx

    for group in ts_analysis['group_combinations']:
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            train_idxs, train_subset = get_group_matches(train_data.data_frame, group, tss.group_by)
            dev_idxs, dev_subset = get_group_matches(dev_data.data_frame, group, tss.group_by)

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

    # afterwards, drop all nans
    # TODO: risk of no valid points...  would have to do this at transform time to solve, not sure if possible!
    train_data.data_frame = train_data.data_frame.dropna()
    dev_data.data_frame = dev_data.data_frame.dropna()

    # TODO: check that the side effects actually worked outside the fn scope


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
    midx = pd.MultiIndex.from_frame(ds.data_frame.reset_index()[[*tss.group_by, 'index']])
    ds.data_frame.index = midx
    ydf.index = midx
    groups = get_ts_groups(ds.data_frame, tss)
    for group in groups:
        group = group if group in ts_analysis['group_combinations'] else '__default'
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            idxs, subset = get_group_matches(ds.data_frame, group, tss.group_by)
            ds.data_frame[target].loc[idxs] = _transform_target(
                ts_analysis, subset[target], group).values
            # TODO: check that the side effects actually worked outside the fn scope and avoid returning?
    return ds.data_frame


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
    for group in groups:
        group = group if group in ts_analysis['group_combinations'] else '__default'
        if len(ts_analysis['group_combinations']) == 1 or group != '__default':
            idxs, subset = get_group_matches(ds.data_frame, group, tss.group_by)
            for timestep in range(tss.horizon):
                ydf[f'prediction_{timestep}'].loc[idxs] = _inverse_transform_target(ts_analysis,
                                                                                    ydf[f'prediction_{timestep}'].loc[
                                                                                        idxs],
                                                                                    group
                                                                                    ).values
    return ydf.reset_index(drop=True)
