from typing import Dict, Tuple
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from lightwood.api.dtype import dtype
from lightwood.ensemble.base import BaseEnsemble
from lightwood.data.encoded_ds import ConcatedEncodedDs
from lightwood.api.types import StatisticalAnalysis, TimeseriesSettings
from lightwood.helpers.ts import add_tn_conf_bounds

from lightwood.analysis.helpers.base import BaseAnalysisBlock
from lightwood.analysis.nc.norm import Normalizer
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.base import CachedRegressorAdapter, CachedClassifierAdapter
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc, RegressorNc, ClassifierNc, MarginErrFunc
from lightwood.analysis.nc.util import clean_df, set_conf_range, get_numerical_conf_range, get_categorical_conf, get_anomalies


"""
Pending:
 - [] simplify nonconformist custom implementation
 - [] reimplement caching for faster analysis?
 - [] confidence for T+N <- active research question
"""


class ICP(BaseAnalysisBlock):
    def analyze(self, info: Dict[str, object]) -> Dict[str, object]:
        # @TODO: move icp_calibration here
        raise NotImplementedError

    def explain(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        # @TODO: move icp_explain here
        raise NotImplementedError


def icp_calibration(
        predictor: BaseEnsemble,
        target: str,
        dtype_dict: dict,
        normal_predictions: pd.DataFrame,
        val_data: pd.DataFrame,
        train_data: pd.DataFrame,
        encoded_val_data: ConcatedEncodedDs,
        is_classification: bool,
        is_numerical: bool,
        is_multi_ts: bool,
        stats_info: StatisticalAnalysis,
        ts_cfg: TimeseriesSettings,
        fixed_significance: float,
        positive_domain: bool,
        confidence_normalizer: bool) -> (Dict, pd.DataFrame):

    """ Confidence estimation with inductive conformal predictors (ICPs) """

    data_type = dtype_dict[target]
    output = {'icp': {'__mdb_active': False}}

    fit_params = {'nr_preds': ts_cfg.nr_predictions or 0, 'columns_to_ignore': []}
    fit_params['columns_to_ignore'].extend([f'timestep_{i}' for i in range(1, fit_params['nr_preds'])])

    if is_classification:
        if predictor.supports_proba:
            all_cat_cols = [col for col in normal_predictions.columns if '__mdb_proba' in col]
            all_classes = np.array([col.replace('__mdb_proba_', '') for col in all_cat_cols])
        else:
            class_keys = sorted(encoded_val_data.encoders[target].rev_map.keys())
            all_classes = np.array([encoded_val_data.encoders[target].rev_map[idx] for idx in class_keys])

        if data_type != dtype.tags:
            enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
            enc.fit(all_classes.reshape(-1, 1))
            output['label_encoders'] = enc  # needed to repr cat labels inside nonconformist
        else:
            output['label_encoders'] = None

        adapter = CachedClassifierAdapter
        nc_function = MarginErrFunc()
        nc_class = ClassifierNc
        icp_class = IcpClassifier

    else:
        adapter = CachedRegressorAdapter
        nc_function = BoostedAbsErrorErrFunc()
        nc_class = RegressorNc
        icp_class = IcpRegressor

    result_df = pd.DataFrame()

    if is_numerical or (is_classification and data_type != dtype.tags):
        model = adapter(predictor)

        norm_params = {'target': target, 'dtype_dict': dtype_dict, 'predictor': predictor,
                       'encoders': encoded_val_data.encoders, 'is_multi_ts': is_multi_ts, 'stop_after': 1e2}
        if confidence_normalizer:
            normalizer = Normalizer(fit_params=norm_params)
            normalizer.fit(train_data)
            normalizer.prediction_cache = normalizer(encoded_val_data)
        else:
            normalizer = None

        # instance the ICP
        nc = nc_class(model, nc_function, normalizer=normalizer)
        icp = icp_class(nc)

        output['icp']['__default'] = icp

        # setup prediction cache to avoid additional .predict() calls
        if is_classification:
            if predictor.models[predictor.best_index].supports_proba:
                icp.nc_function.model.prediction_cache = normal_predictions[all_cat_cols].values
            else:
                predicted_classes = pd.get_dummies(normal_predictions['prediction']).values  # inflate to one-hot enc
                icp.nc_function.model.prediction_cache = predicted_classes

        elif is_multi_ts:
            # we fit ICPs for time series confidence bounds only at t+1 forecast
            icp.nc_function.model.prediction_cache = np.array([p[0] for p in normal_predictions['prediction']])
        else:
            icp.nc_function.model.prediction_cache = np.array(normal_predictions['prediction'])

        if not is_classification:
            output['df_std_dev'] = {'__default': stats_info.df_std_dev}

        # fit additional ICPs in time series tasks with grouped columns
        if ts_cfg.is_timeseries and ts_cfg.group_by:

            # create an ICP for each possible group
            group_info = val_data[ts_cfg.group_by].to_dict('list')
            all_group_combinations = list(product(*[set(x) for x in group_info.values()]))
            output['icp']['__mdb_groups'] = all_group_combinations
            output['icp']['__mdb_group_keys'] = [x for x in group_info.keys()]

            for combination in all_group_combinations:
                output['icp'][frozenset(combination)] = deepcopy(icp)

        # calibrate ICP
        icp_df = deepcopy(val_data)
        icp_df, y = clean_df(icp_df, target, is_classification, output.get('label_encoders', None))
        output['icp']['__default'].index = icp_df.columns
        output['icp']['__default'].calibrate(icp_df.values, y)

        # get confidence estimation for validation dataset
        conf, ranges = set_conf_range(
            icp_df, icp, dtype_dict[target],
            output, positive_domain=positive_domain, significance=fixed_significance)
        if not is_classification:
            result_df = pd.DataFrame(index=val_data.index, columns=['confidence', 'lower', 'upper'], dtype=float)
            result_df.loc[icp_df.index, 'lower'] = ranges[:, 0]
            result_df.loc[icp_df.index, 'upper'] = ranges[:, 1]
        else:
            result_df = pd.DataFrame(index=val_data.index, columns=['confidence'], dtype=float)

        result_df.loc[icp_df.index, 'confidence'] = conf

        # calibrate additional grouped ICPs
        if ts_cfg.is_timeseries and ts_cfg.group_by:
            icps = output['icp']
            group_keys = icps['__mdb_group_keys']

            # add all predictions to DF
            icps_df = deepcopy(val_data)
            if is_multi_ts:
                icps_df[f'__predicted_{target}'] = [p[0] for p in normal_predictions['prediction']]
            else:
                icps_df[f'__predicted_{target}'] = normal_predictions['prediction']

            for group in icps['__mdb_groups']:
                icp_df = icps_df
                if icps[frozenset(group)].nc_function.normalizer is not None:
                    icp_df[f'__norm_{target}'] = icps[frozenset(group)].nc_function.normalizer.prediction_cache

                # filter irrelevant rows for each group combination
                for key, val in zip(group_keys, group):
                    icp_df = icp_df[icp_df[key] == val]

                # save relevant predictions in the caches, then calibrate the ICP
                pred_cache = icp_df.pop(f'__predicted_{target}').values
                icps[frozenset(group)].nc_function.model.prediction_cache = pred_cache
                icp_df, y = clean_df(icp_df, target, is_classification, output.get('label_encoders', None))
                if icps[frozenset(group)].nc_function.normalizer is not None:
                    icps[frozenset(group)].nc_function.normalizer.prediction_cache = icp_df.pop(
                        f'__norm_{target}').values

                icps[frozenset(group)].index = icp_df.columns  # important at inference time
                icps[frozenset(group)].calibrate(icp_df.values, y)

                # save training std() for bounds width selection
                if not is_classification:
                    icp_train_df = val_data
                    for key, val in zip(group_keys, group):
                        icp_train_df = icp_train_df[icp_train_df[key] == val]
                    y_train = icp_train_df[target].values
                    output['df_std_dev'][frozenset(group)] = y_train.std()

                # get bounds for relevant rows in validation dataset
                conf, group_ranges = set_conf_range(
                    icp_df, icps[frozenset(group)],
                    dtype_dict[target],
                    output, group=frozenset(group),
                    positive_domain=positive_domain, significance=fixed_significance)
                # save group bounds
                if not is_classification:
                    result_df.loc[icp_df.index, 'lower'] = group_ranges[:, 0]
                    result_df.loc[icp_df.index, 'upper'] = group_ranges[:, 1]

                result_df.loc[icp_df.index, 'confidence'] = conf

        # consolidate all groups here
        output['icp']['__mdb_active'] = True

    return output, result_df


def icp_explain(data,
                encoded_data,
                predictions,
                analysis: Dict,
                insights: pd.DataFrame,
                target_name: str,
                target_dtype: str,
                tss: TimeseriesSettings,
                positive_domain: bool,
                fixed_confidence: float,
                anomaly_detection: bool,
                anomaly_error_rate: float,
                anomaly_cooldown: int) -> pd.DataFrame:

    icp_X = deepcopy(data)

    # replace observed data w/predictions
    preds = predictions['prediction']
    if tss.is_timeseries and tss.nr_predictions > 1:
        preds = [p[0] for p in preds]

        for col in [f'timestep_{i}' for i in range(1, tss.nr_predictions)]:
            if col in icp_X.columns:
                icp_X.pop(col)  # erase ignorable columns

    icp_X[target_name] = preds

    is_categorical = target_dtype in (dtype.binary, dtype.categorical, dtype.array)
    is_numerical = target_dtype in [dtype.integer, dtype.float] or target_dtype == dtype.array
    is_anomaly_task = is_numerical and tss.is_timeseries and anomaly_detection

    if (is_numerical or is_categorical) and analysis['icp'].get('__mdb_active', False):

        # reorder DF index
        index = analysis['icp']['__default'].index.values
        index = np.append(index, target_name) if target_name not in index else index
        icp_X = icp_X.reindex(columns=index)  # important, else bounds can be invalid

        # only one normalizer, even if it's a grouped time series task
        normalizer = analysis['icp']['__default'].nc_function.normalizer
        if normalizer:
            normalizer.prediction_cache = normalizer(encoded_data)
            icp_X['__mdb_selfaware_scores'] = normalizer.prediction_cache

        # get ICP predictions
        result_cols = ['lower', 'upper', 'significance'] if is_numerical else ['significance']
        result = pd.DataFrame(index=icp_X.index, columns=result_cols)

        # base ICP
        X = deepcopy(icp_X)
        # Calling `values` multiple times increased runtime of this function; referenced var is faster
        icp_values = X.values

        # get all possible ranges
        if tss.is_timeseries and tss.nr_predictions > 1 and is_numerical:

            # bounds in time series are only given for the first forecast
            analysis['icp']['__default'].nc_function.model.prediction_cache = \
                [p[0] for p in predictions['prediction']]
            all_confs = analysis['icp']['__default'].predict(icp_values)

        elif is_numerical:
            analysis['icp']['__default'].nc_function.model.prediction_cache = predictions['prediction']
            all_confs = analysis['icp']['__default'].predict(icp_values)

        # categorical
        else:
            predicted_proba = True if any(['__mdb_proba' in col for col in predictions.columns]) else False
            if predicted_proba:
                all_cat_cols = [col for col in predictions.columns if '__mdb_proba' in col]
                class_dists = predictions[all_cat_cols].values
                for icol, cat_col in enumerate(all_cat_cols):
                    insights.loc[X.index, cat_col] = class_dists[:, icol]
            else:
                class_dists = pd.get_dummies(predictions['prediction']).values

            analysis['icp']['__default'].nc_function.model.prediction_cache = class_dists

            conf_candidates = list(range(20)) + list(range(20, 100, 10))
            all_ranges = np.array(
                [analysis['icp']['__default'].predict(icp_values, significance=s / 100)
                 for s in conf_candidates])
            all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

        # convert (B, 2, 99) into (B, 2) given width or error rate constraints
        if is_numerical:
            significances = fixed_confidence
            if significances is not None:
                confs = all_confs[:, :, int(100 * (1 - significances)) - 1]
            else:
                error_rate = anomaly_error_rate if is_anomaly_task else None
                significances, confs = get_numerical_conf_range(all_confs,
                                                                df_std_dev=analysis['df_std_dev'],
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
                                                                            df_std_dev=analysis['df_std_dev'],
                                                                            positive_domain=positive_domain,
                                                                            group=frozenset(group),
                                                                            error_rate=error_rate)

                            # only replace where grouped ICP is more informative (i.e. tighter)
                            default_icp_widths = result.loc[X.index, 'upper'] - result.loc[X.index, 'lower']
                            grouped_widths = np.subtract(confs[:, 1], confs[:, 0])
                            insert_index = (default_icp_widths > grouped_widths)[lambda x: x.isin([True])].index
                            conf_index = (default_icp_widths.reset_index(drop=True) >
                                          grouped_widths)[lambda x: x.isin([True])].index

                            result.loc[insert_index, 'lower'] = confs[conf_index, 0]
                            result.loc[insert_index, 'upper'] = confs[conf_index, 1]
                            result.loc[insert_index, 'significance'] = significances[conf_index]

                        else:
                            conf_candidates = list(range(20)) + list(range(20, 100, 10))
                            all_ranges = np.array(
                                [icp.predict(X.values, significance=s / 100)
                                 for s in conf_candidates])
                            all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)
                            significances = get_categorical_conf(all_confs, conf_candidates)
                            result.loc[X.index, 'significance'] = significances

        insights['confidence'] = result['significance'].astype(float).tolist()

        if is_numerical:
            insights['lower'] = result['lower'].astype(float)
            insights['upper'] = result['upper'].astype(float)

        # anomaly detection
        if is_anomaly_task:
            anomalies = get_anomalies(insights,
                                      data[target_name],
                                      cooldown=anomaly_cooldown)
            insights['anomaly'] = anomalies

        if tss.is_timeseries and tss.nr_predictions > 1 and is_numerical:
            insights = add_tn_conf_bounds(insights, tss)

    return insights