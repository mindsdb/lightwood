from mindsdb_native.libs.helpers.confidence_helpers import get_numerical_conf_range, get_categorical_conf, get_anomalies
from mindsdb_native.libs.helpers.conformal_helpers import restore_icp_state, clear_icp_state

from copy import deepcopy
import pandas as pd
import numpy as np


def explain():

    output_data = {col: [] for col in lmd['columns']}
    predictions_df = input_data.data_frame

    for col in lmd['columns']:
        if col in lmd['predict_columns']:
            output_data[f'__observed_{col}'] = list(predictions_df[col])
            output_data[col] = hmd['predictions'][col]

            if f'{col}_class_distribution' in hmd['predictions']:
                output_data[f'{col}_class_distribution'] = hmd['predictions'][f'{col}_class_distribution']
                lmd['lightwood_data'][f'{col}_class_map'] = lmd['stats_v2'][col]['lightwood_class_map']
        else:
            output_data[col] = list(predictions_df[col])

    # confidence estimation using calibrated inductive conformal predictors (ICPs)
    if hmd['icp']['__mdb_active'] and not lmd['quick_predict']:

        icp_X = deepcopy(input_data.cached_pred_df)

        # replace observed data w/predictions
        for col in lmd['predict_columns']:
            if col in icp_X.columns:
                preds = hmd['predictions'][col]
                if lmd['tss']['is_timeseries'] and lmd['tss']['nr_predictions'] > 1:
                    preds = [p[0] for p in preds]
                icp_X[col] = preds

        # erase ignorable columns
        for col in lmd['columns_to_ignore']:
            if col in icp_X.columns:
                icp_X.pop(col)

        # get confidence bounds for each target
        for predicted_col in lmd['predict_columns']:
            output_data[f'{predicted_col}_confidence'] = [None] * len(output_data[predicted_col])
            output_data[f'{predicted_col}_confidence_range'] = [[None, None]] * len(output_data[predicted_col])

            typing_info = lmd['stats_v2'][predicted_col]['typing']
            is_numerical = typing_info['data_type'] == DATA_TYPES.NUMERIC or \
                           (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                            DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys())

            is_categorical = (typing_info['data_type'] == DATA_TYPES.CATEGORICAL or
                             (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                              DATA_TYPES.CATEGORICAL in typing_info['data_type_dist'].keys())) and \
                              typing_info['data_subtype'] != DATA_SUBTYPES.TAGS

            is_anomaly_task = is_numerical and \
                              lmd['tss']['is_timeseries'] and \
                              lmd['anomaly_detection']

            if (is_numerical or is_categorical) and hmd['icp'].get(predicted_col, False):

                # reorder DF index
                index = hmd['icp'][predicted_col]['__default'].index.values
                index = np.append(index, predicted_col) if predicted_col not in index else index
                icp_X = icp_X.reindex(columns=index)  # important, else bounds can be invalid

                # only one normalizer, even if it's a grouped time series task
                normalizer = hmd['icp'][predicted_col]['__default'].nc_function.normalizer
                if normalizer:
                    normalizer.prediction_cache = hmd['predictions'].get(f'{predicted_col}_selfaware_scores',
                                                                              None)
                    icp_X['__mdb_selfaware_scores'] = normalizer.prediction_cache

                # get ICP predictions
                result = pd.DataFrame(index=icp_X.index, columns=['lower', 'upper', 'significance'])

                # base ICP
                X = deepcopy(icp_X)

                # get all possible ranges
                if lmd['tss']['is_timeseries'] and lmd['tss']['nr_predictions'] > 1 and is_numerical:

                    # bounds in time series are only given for the first forecast
                    hmd['icp'][predicted_col]['__default'].nc_function.model.prediction_cache = \
                        [p[0] for p in output_data[predicted_col]]
                    all_confs = hmd['icp'][predicted_col]['__default'].predict(X.values)

                elif is_numerical:
                    hmd['icp'][predicted_col]['__default'].nc_function.model.prediction_cache = \
                        output_data[predicted_col]
                    all_confs = hmd['icp'][predicted_col]['__default'].predict(X.values)

                # categorical
                else:
                    hmd['icp'][predicted_col]['__default'].nc_function.model.prediction_cache = \
                        output_data[f'{predicted_col}_class_distribution']

                    conf_candidates = list(range(20)) + list(range(20, 100, 10))
                    all_ranges = np.array(
                        [hmd['icp'][predicted_col]['__default'].predict(X.values, significance=s / 100)
                         for s in conf_candidates])
                    all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

                # convert (B, 2, 99) into (B, 2) given width or error rate constraints
                if is_numerical:
                    significances = lmd.get('fixed_confidence', None)
                    if significances is not None:
                        confs = all_confs[:, :, int(100*(1-significances))-1]
                    else:
                        error_rate = lmd['anomaly_error_rate'] if is_anomaly_task else None
                        significances, confs = get_numerical_conf_range(all_confs,
                                                                        predicted_col,
                                                                        lmd['stats_v2'],
                                                                        error_rate=error_rate)
                    result.loc[X.index, 'lower'] = confs[:, 0]
                    result.loc[X.index, 'upper'] = confs[:, 1]
                else:
                    conf_candidates = list(range(20)) + list(range(20, 100, 10))
                    significances = get_categorical_conf(all_confs, conf_candidates)

                result.loc[X.index, 'significance'] = significances

                # grouped time series, we replace bounds in rows that have a trained ICP
                if hmd['icp'][predicted_col].get('__mdb_groups', False):
                    icps = hmd['icp'][predicted_col]
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
                                icp.nc_function.model.prediction_cache = X.pop(predicted_col).values
                                if icp.nc_function.normalizer:
                                    icp.nc_function.normalizer.prediction_cache = X.pop('__mdb_selfaware_scores').values

                                # predict and get confidence level given width or error rate constraints
                                if is_numerical:
                                    all_confs = icp.predict(X.values)
                                    error_rate = lmd['anomaly_error_rate'] if is_anomaly_task else None
                                    significances, confs = get_numerical_conf_range(all_confs, predicted_col,
                                                                                    lmd['stats_v2'],
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

                output_data[f'{predicted_col}_confidence'] = result['significance'].tolist()
                confs = [[a, b] for a, b in zip(result['lower'], result['upper'])]
                output_data[f'{predicted_col}_confidence_range'] = confs

                # anomaly detection
                if is_anomaly_task:
                    anomalies = get_anomalies(output_data[f'{predicted_col}_confidence_range'],
                                              output_data[f'__observed_{predicted_col}'],
                                              cooldown=lmd['anomaly_cooldown'])
                    output_data[f'{predicted_col}_anomaly'] = anomalies

    else:
        for predicted_col in lmd['predict_columns']:
            output_data[f'{predicted_col}_confidence'] = [None] * len(output_data[predicted_col])
            output_data[f'{predicted_col}_confidence_range'] = [[None, None]] * len(output_data[predicted_col])

    output_data = PredictTransactionOutputData(
        transaction=self,
        data=output_data
    )

    return output_data