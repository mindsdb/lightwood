from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from lightwood.analysis.nc.util import get_numerical_conf_range, get_categorical_conf, get_anomalies  #  restore_icp_state, clear_icp_state
from lightwood.helpers.ts import get_inferred_timestamps, add_tn_conf_bounds
from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings


def explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            timeseries_settings: TimeseriesSettings,
            analysis: dict,
            target_name: str,
            target_dtype: str,
            positive_domain: bool,
            fixed_confidence: float,
            anomaly_detection: bool,

            # forces specific confidence level in ICP
            anomaly_error_rate: float,

            # ignores anomaly detection for N steps after an
            # initial anomaly triggers the cooldown period;
            # implicitly assumes series are regularly spaced
            anomaly_cooldown: int,

            ts_analysis: dict = None
            ):

    # @TODO: check not quick_predict
    data = data.reset_index(drop=True)

    insights = pd.DataFrame()
    insights['truth'] = data[target_name]
    insights['prediction'] = predictions['prediction']

    if timeseries_settings.is_timeseries:

        if timeseries_settings.group_by:
            for col in timeseries_settings.group_by:
                insights[f'group_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            insights[f'order_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            insights[f'order_{col}'] = get_inferred_timestamps(insights, col, ts_analysis['deltas'], timeseries_settings)

    # confidence estimation using calibrated inductive conformal predictors (ICPs)
    if analysis['icp']['__mdb_active']:
        icp_X = deepcopy(data)

        # replace observed data w/predictions
        preds = predictions['prediction']
        if timeseries_settings.is_timeseries and timeseries_settings.nr_predictions > 1:
            preds = [p[0] for p in preds]

        icp_X[target_name] = preds

        # erase ignorable columns @TODO: reintroduce?
        # for col in pdef['columns_to_ignore']:
        #     if col in icp_X.columns:
        #         icp_X.pop(col)

        is_categorical = target_dtype in (dtype.binary, dtype.categorical, dtype.array)
        is_numerical = target_dtype in [dtype.integer, dtype.float] or target_dtype == dtype.array
        is_anomaly_task = is_numerical and timeseries_settings.is_timeseries and anomaly_detection

        if (is_numerical or is_categorical) and analysis['icp'].get('__mdb_active', False):

            # reorder DF index
            index = analysis['icp']['__default'].index.values
            index = np.append(index, target_name) if target_name not in index else index
            icp_X = icp_X.reindex(columns=index)  # important, else bounds can be invalid

            # only one normalizer, even if it's a grouped time series task
            normalizer = analysis['icp']['__default'].nc_function.normalizer
            if normalizer:
                normalizer.prediction_cache = normalizer.predict(encoded_data)
                icp_X['__mdb_selfaware_scores'] = normalizer.prediction_cache

            # get ICP predictions
            result = pd.DataFrame(index=icp_X.index, columns=['lower', 'upper', 'significance'])

            # base ICP
            X = deepcopy(icp_X)

            # get all possible ranges
            if timeseries_settings.is_timeseries and timeseries_settings.nr_predictions > 1 and is_numerical:

                # bounds in time series are only given for the first forecast
                analysis['icp']['__default'].nc_function.model.prediction_cache = \
                    [p[0] for p in predictions['prediction']]
                all_confs = analysis['icp']['__default'].predict(X.values)

            elif is_numerical:
                analysis['icp']['__default'].nc_function.model.prediction_cache = predictions['prediction']
                all_confs = analysis['icp']['__default'].predict(X.values)

            # categorical
            else:
                # @TODO use the real target_class_distribution
                class_dists = pd.get_dummies(predictions['prediction']).values
                analysis['icp']['__default'].nc_function.model.prediction_cache = class_dists

                conf_candidates = list(range(20)) + list(range(20, 100, 10))
                all_ranges = np.array(
                    [analysis['icp']['__default'].predict(X.values, significance=s / 100)
                     for s in conf_candidates])
                all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

            # convert (B, 2, 99) into (B, 2) given width or error rate constraints
            if is_numerical:
                significances = fixed_confidence
                if significances is not None:
                    confs = all_confs[:, :, int(100*(1-significances))-1]
                else:
                    error_rate = anomaly_error_rate if is_anomaly_task else None
                    significances, confs = get_numerical_conf_range(all_confs,
                                                                    train_std_dev=analysis['train_std_dev'],
                                                                    positive_domain=positive_domain,
                                                                    error_rate=error_rate)
                result.loc[X.index, 'lower'] = confs[:, 0]
                result.loc[X.index, 'upper'] = confs[:, 1]
            else:
                conf_candidates = list(range(20)) + list(range(20, 100, 10))
                significances = get_categorical_conf(all_confs, conf_candidates)

            result.loc[X.index, 'significance'] = significances

            # grouped time series, we replace bounds in rows that have a trained ICP
            if analysis['icp'].get('__mdb_groups', False):
                icps = analysis['icp']
                group_keys = icps['__mdb_group_keys']

                for group in icps['__mdb_groups']:
                    icp = icps[frozenset(group)]

                    # check ICP has calibration scores
                    if icp.cal_scores[0].shape[0] > 0:

                        # filter rows by group
                        X = deepcopy(icp_X)
                        for key, val in zip(group_keys, group):
                            X = X[X[key] == val]

                        if X.size > 0:
                            # set ICP caches
                            icp.nc_function.model.prediction_cache = X.pop(target_name).values
                            if icp.nc_function.normalizer:
                                icp.nc_function.normalizer.prediction_cache = X.pop('__mdb_selfaware_scores').values

                            # predict and get confidence level given width or error rate constraints
                            if is_numerical:
                                all_confs = icp.predict(X.values)
                                error_rate = anomaly_error_rate if is_anomaly_task else None
                                significances, confs = get_numerical_conf_range(all_confs,
                                                                                train_std_dev=analysis['train_std_dev'],
                                                                                positive_domain=positive_domain,
                                                                                group=frozenset(group),
                                                                                error_rate=error_rate)

                                # only replace where grouped ICP is more informative (i.e. tighter)
                                default_icp_widths = result.loc[X.index, 'upper'] - result.loc[X.index, 'lower']
                                grouped_widths = np.subtract(confs[:, 1], confs[:, 0])
                                insert_index = (default_icp_widths > grouped_widths)[lambda x: x==True].index

                                result.loc[insert_index, 'lower'] = confs[:, 0]
                                result.loc[insert_index, 'upper'] = confs[:, 1]
                                result.loc[insert_index, 'significance'] = significances

                            else:
                                conf_candidates = list(range(20)) + list(range(20, 100, 10))
                                all_ranges = np.array(
                                    [icp.predict(X.values, significance=s / 100)
                                     for s in conf_candidates])
                                all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)
                                significances = get_categorical_conf(all_confs, conf_candidates)
                                result.loc[X.index, 'significance'] = significances

            insights['confidence'] = result['significance'].astype(float).tolist()
            insights['lower'] = result['lower'].astype(float)
            insights['upper'] = result['upper'].astype(float)

            # anomaly detection
            if is_anomaly_task:
                anomalies = get_anomalies(insights,
                                          data[target_name],
                                          cooldown=anomaly_cooldown)
                insights['anomaly'] = anomalies

            # @TODO: add T+N confidence bounds and disaggregate into rows if nr_predictions > 1
            if timeseries_settings.is_timeseries and timeseries_settings.nr_predictions > 1 and is_numerical:
                insights = add_tn_conf_bounds(insights, timeseries_settings)

    return insights