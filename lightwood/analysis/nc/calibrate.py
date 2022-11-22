import inspect
from copy import deepcopy
from typing import Dict, Tuple, Optional
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from type_infer.dtype import dtype
from lightwood.api.types import PredictionArguments
from lightwood.helpers.ts import add_tn_num_conf_bounds, add_tn_cat_conf_bounds, get_ts_groups

from lightwood.data import EncodedDs
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.analysis.nc.norm import Normalizer
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier, IcpTSRegressor
from lightwood.analysis.nc.base import CachedRegressorAdapter, CachedClassifierAdapter, CachedTSAdapter
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc, RegressorNc, ClassifierNc, MarginErrFunc, TSNc, \
    TSAbsErrorErrFunc
from lightwood.analysis.nc.util import clean_df, set_conf_range, get_numeric_conf_range, \
    get_categorical_conf, get_anomalies, get_ts_conf_range


class ICP(BaseAnalysisBlock):
    """ Confidence estimation block, uses inductive conformal predictors (ICPs) for model agnosticity """

    def __init__(self,
                 confidence_normalizer: Optional[bool] = False,
                 fixed_significance: Optional[float] = None,
                 deps: Optional[tuple] = tuple()
                 ):
        super().__init__(deps=deps)
        self.fixed_significance = fixed_significance
        self.confidence_normalizer = confidence_normalizer
        self.validation_size = 100  # determines size of nonconformity score arrays (has sizable impact in runtime)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        data_type = ns.dtype_dict[ns.target]
        output = {'icp': {'__mdb_active': False}}

        if 'confidence' in ns.normal_predictions.columns:
            #  bypass calibrator if model already outputs confidence
            output['result_df'] = ns.normal_predictions[['confidence', 'lower', 'upper']]
            return {**info, **output}

        fit_params = {'horizon': ns.tss.horizon or 0, 'columns_to_ignore': []}
        fit_params['columns_to_ignore'].extend([f'timestep_{i}' for i in range(1, fit_params['horizon'])])

        if ns.is_classification:
            if ns.predictor.supports_proba:
                all_cat_cols = [col for col in ns.normal_predictions.columns
                                if '__mdb_proba' in col and '__mdb_unknown_cat' not in col]
                all_classes = np.array([col.replace('__mdb_proba_', '') for col in all_cat_cols])
            else:
                class_keys = sorted(ns.encoded_val_data.encoders[ns.target].rev_map.keys())
                all_classes = np.array([ns.encoded_val_data.encoders[ns.target].rev_map[idx] for idx in class_keys])

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
        elif ns.is_multi_ts:
            adapter = CachedTSAdapter
            nc_function = TSAbsErrorErrFunc(horizon_length=ns.tss.horizon)
            nc_class = TSNc
            icp_class = IcpTSRegressor
        else:
            adapter = CachedRegressorAdapter
            nc_function = BoostedAbsErrorErrFunc()
            nc_class = RegressorNc
            icp_class = IcpRegressor

        result_df = pd.DataFrame()

        if ns.is_numerical or (ns.is_classification and data_type != dtype.tags):
            model = adapter(ns.predictor)

            norm_params = {'target': ns.target, 'dtype_dict': ns.dtype_dict, 'predictor': ns.predictor,
                           'encoders': ns.encoded_val_data.encoders, 'is_multi_ts': ns.is_multi_ts, 'stop_after': 1e2}
            if self.confidence_normalizer:
                normalizer = Normalizer(fit_params=norm_params)
                normalizer.fit(ns.train_data)
                normalizer.prediction_cache = normalizer(ns.encoded_val_data, args=PredictionArguments())
            else:
                normalizer = None

            # instance the ICP
            nc = nc_class(model, nc_function, normalizer=normalizer)
            if 'horizon_length' in inspect.signature(icp_class).parameters:
                icp = icp_class(nc, horizon_length=ns.tss.horizon, cal_size=self.validation_size)
            else:
                icp = icp_class(nc, cal_size=self.validation_size)

            output['icp']['__default'] = icp
            icp_df = deepcopy(ns.data)

            # setup prediction cache to avoid additional .predict() calls
            pred_is_list = isinstance(ns.normal_predictions['prediction'], list) and \
                isinstance(ns.normal_predictions['prediction'][0], list)
            if ns.is_classification:
                if ns.predictor.supports_proba:
                    icp.nc_function.model.prediction_cache = ns.normal_predictions[all_cat_cols].values
                else:
                    if ns.is_multi_ts:
                        icp.nc_function.model.prediction_cache = np.array(
                            [p[0] for p in ns.normal_predictions['prediction']])
                        preds = icp.nc_function.model.prediction_cache
                    else:
                        preds = ns.normal_predictions['prediction']
                    predicted_classes = pd.get_dummies(preds).values  # inflate to one-hot enc
                    icp.nc_function.model.prediction_cache = predicted_classes

            elif ns.is_multi_ts or pred_is_list:
                icp.nc_function.model.prediction_cache = np.array(
                    [np.array(p) for p in ns.normal_predictions['prediction']])
            else:
                icp.nc_function.model.prediction_cache = np.array(ns.normal_predictions['prediction'])

            if not ns.is_classification:
                output['df_target_stddev'] = {'__default': ns.stats_info.df_target_stddev}

            # fit additional ICPs in time series tasks with grouped columns
            if ns.tss.is_timeseries and ns.tss.group_by:
                # generate a multiindex
                midx = pd.MultiIndex.from_frame(icp_df[[*ns.tss.group_by, f'__mdb_original_{ns.tss.order_by}']])
                icp_df.index = midx
                result_df.index = midx

                # create an ICP for each possible group
                group_info = ns.data[ns.tss.group_by].to_dict('list')
                all_group_combinations = get_ts_groups(ns.data, ns.tss)
                all_group_combinations.remove('__default')
                output['icp']['__mdb_groups'] = all_group_combinations
                output['icp']['__mdb_group_keys'] = [x for x in group_info.keys()]

                for combination in all_group_combinations:
                    output['icp'][tuple(combination)] = deepcopy(icp)

            # calibrate ICP
            icp_df = deepcopy(ns.data)
            icp_df, y = clean_df(icp_df, ns, output.get('label_encoders', None))
            if ns.tss.is_timeseries and ns.tss.group_by:
                icp_df.index = midx
            output['icp']['__default'].index = icp_df.columns
            output['icp']['__default'].calibrate(icp_df.values, y)

            # get confidence estimation for validation dataset
            conf, ranges = set_conf_range(
                icp_df, icp, ns.dtype_dict[ns.target],
                output, positive_domain=ns.stats_info.positive_domain, significance=self.fixed_significance)
            if not ns.is_classification:
                result_df = pd.DataFrame(index=icp_df.index, columns=['confidence', 'lower', 'upper'], dtype=float)
                result_df.loc[icp_df.index, 'lower'] = ranges[:, 0]
                result_df.loc[icp_df.index, 'upper'] = ranges[:, 1]
            else:
                result_df = pd.DataFrame(index=icp_df.index, columns=['confidence'], dtype=float)

            result_df.loc[icp_df.index, 'confidence'] = conf

            # calibrate additional grouped ICPs
            if ns.tss.is_timeseries and ns.tss.group_by:
                icps = output['icp']
                group_keys = icps['__mdb_group_keys']

                # add all predictions to DF
                icps_df = deepcopy(ns.data)
                midx = pd.MultiIndex.from_frame(icps_df[[*ns.tss.group_by, f'__mdb_original_{ns.tss.order_by}']])
                icps_df.index = midx
                if ns.is_multi_ts or pred_is_list:
                    icps_df[f'__predicted_{ns.target}'] = [np.array(p) for p in ns.normal_predictions['prediction']]
                else:
                    icps_df[f'__predicted_{ns.target}'] = np.array(ns.normal_predictions['prediction'])

                for group in icps['__mdb_groups']:
                    icp_df = icps_df

                    # filter irrelevant rows for each group combination
                    icp_df['__mdb_norm_index'] = np.arange(len(icp_df))
                    for key, val in zip(group_keys, group):
                        icp_df = icp_df[icp_df[key] == val]

                    if icps[tuple(group)].nc_function.normalizer is not None:
                        group_normalizer = icps[tuple(group)].nc_function.normalizer
                        norm_input_df = ns.encoded_val_data.data_frame.iloc[icp_df.pop('__mdb_norm_index')]
                        norm_input = EncodedDs(ns.encoded_val_data.encoders, norm_input_df, ns.target)
                        norm_cache = group_normalizer(norm_input, args=PredictionArguments())
                        icp_df[f'__norm_{ns.target}'] = norm_cache

                    # save relevant predictions in the caches, then calibrate the ICP
                    pred_cache = icp_df.pop(f'__predicted_{ns.target}').values
                    if ns.is_multi_ts:
                        pred_cache = np.array([np.array(p) for p in pred_cache])
                    icps[tuple(group)].nc_function.model.prediction_cache = pred_cache
                    icp_df, y = clean_df(icp_df, ns, output.get('label_encoders', None))
                    if icps[tuple(group)].nc_function.normalizer is not None:
                        icps[tuple(group)].nc_function.normalizer.prediction_cache = icp_df.pop(
                            f'__norm_{ns.target}').values

                    icps[tuple(group)].index = icp_df.columns  # important at inference time
                    icps[tuple(group)].calibrate(icp_df.values, y)

                    # save training std() for bounds width selection
                    if not ns.is_classification:
                        icp_train_df = ns.data
                        for key, val in zip(group_keys, group):
                            icp_train_df = icp_train_df[icp_train_df[key] == val]
                        y_train = icp_train_df[ns.target].values
                        output['df_target_stddev'][tuple(group)] = y_train.std()

                    # get bounds for relevant rows in validation dataset
                    conf, group_ranges = set_conf_range(
                        icp_df, icps[tuple(group)],
                        ns.dtype_dict[ns.target],
                        output, group=tuple(group),
                        positive_domain=ns.stats_info.positive_domain, significance=self.fixed_significance)
                    # save group bounds
                    if not ns.is_classification:
                        result_df.loc[icp_df.index, 'lower'] = group_ranges[:, 0]
                        result_df.loc[icp_df.index, 'upper'] = group_ranges[:, 1]

                    result_df.loc[icp_df.index, 'confidence'] = conf

            # consolidate all groups here
            output['icp']['__mdb_active'] = True

        result_df.index = ns.data.index
        output['result_df'] = result_df

        info = {**info, **output}
        return info

    def explain(self, row_insights: pd.DataFrame, global_insights: Dict[str, object],
                **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        ns = SimpleNamespace(**kwargs)

        is_categorical = ns.target_dtype in (dtype.binary, dtype.categorical, dtype.cat_array, dtype.cat_tsarray)
        is_numerical = ns.target_dtype in (dtype.integer, dtype.float,
                                           dtype.quantity, dtype.num_array, dtype.num_tsarray)
        is_multi_ts = ns.tss.is_timeseries and ns.tss.horizon > 1
        is_anomaly_task = is_numerical and ns.tss.is_timeseries and ns.anomaly_detection

        if 'confidence' in ns.predictions.columns:
            # bypass calibrator if model already outputs confidence
            row_insights['prediction'] = ns.predictions['prediction']
            if 'upper' in ns.predictions.columns and 'lower' in ns.predictions.columns:
                row_insights['upper'] = ns.predictions['upper']
                row_insights['lower'] = ns.predictions['lower']

            if not isinstance(ns.predictions['confidence'].iloc[0], list) and ns.tss.horizon > 1:
                row_insights['confidence'] = ns.predictions['confidence'].astype(object)
                row_insights['confidence'] = row_insights['confidence'].apply(
                    lambda x: [x for _ in range(ns.tss.horizon)])
            else:
                row_insights['confidence'] = ns.predictions['confidence']
            return self._formatted(row_insights, global_insights, ns, is_numerical)

        if ns.analysis['icp']['__mdb_active']:
            icp_X = deepcopy(ns.data)

            # replace observed data w/predictions
            preds = ns.predictions['prediction']
            if is_multi_ts and is_numerical:
                preds = np.array([np.array(p) for p in preds])

                for col in [f'timestep_{i}' for i in range(1, ns.tss.horizon)]:
                    if col in icp_X.columns:
                        icp_X.pop(col)  # erase ignorable columns

                target_cols = [ns.target_name] + [f'{ns.target_name}_timestep_{i}' for i in range(1, ns.tss.horizon)]
                icp_X[target_cols] = preds
            elif is_multi_ts and is_categorical:
                preds = [p[0] for p in preds]
                icp_X[ns.target_name] = preds
            else:
                icp_X[ns.target_name] = preds

            if (is_numerical or is_categorical) and ns.analysis['icp'].get('__mdb_active', False):
                base_icp = ns.analysis['icp']['__default']
                # reorder DF index
                index = base_icp.index.values
                if ns.target_name not in index:
                    if is_multi_ts:
                        index = np.array(list(index) + [ns.target_name] +
                                         [f'{ns.target_name}_timestep_{i}' for i in range(1, ns.tss.horizon)])
                    else:
                        index = np.append(index, ns.target_name)
                icp_X = icp_X.reindex(columns=index)  # important, else bounds can be invalid

                # only one normalizer, even if it's a grouped time series task
                normalizer = base_icp.nc_function.normalizer
                if normalizer:
                    normalizer.prediction_cache = normalizer(ns.encoded_data, args=PredictionArguments)
                    icp_X['__mdb_selfaware_scores'] = normalizer.prediction_cache

                # get ICP predictions
                if is_multi_ts:
                    result_cols = ['significance', 'lower', 'upper', ] + \
                                  [f'lower_timestep_{i}' for i in range(1, ns.tss.horizon)] + \
                                  [f'upper_timestep_{i}' for i in range(1, ns.tss.horizon)] + \
                                  [f'significance_timestep_{i}' for i in range(1, ns.tss.horizon)]
                elif is_numerical:
                    result_cols = ['lower', 'upper', 'significance']
                else:
                    result_cols = ['significance']
                result = pd.DataFrame(index=icp_X.index, columns=result_cols)

                # base ICP
                X = deepcopy(icp_X)
                # Calling `values` multiple times increased runtime of this function; referenced var is faster
                icp_values = X.values

                # get all possible ranges
                if is_numerical:
                    base_icp.nc_function.model.prediction_cache = preds
                    all_confs = base_icp.predict(icp_values)

                # categorical
                else:
                    predicted_proba = True if any(['__mdb_proba' in col for col in ns.predictions.columns]) else False
                    if predicted_proba:
                        all_cat_cols = [col for col in ns.predictions.columns
                                        if '__mdb_proba' in col and '__mdb_unknown_cat' not in col]
                        class_dists = ns.predictions[all_cat_cols].values
                        for icol, cat_col in enumerate(all_cat_cols):
                            row_insights.loc[X.index, cat_col] = class_dists[:, icol]
                    else:
                        class_dists = pd.get_dummies(preds).values

                    base_icp.nc_function.model.prediction_cache = class_dists

                    all_ranges = np.array([base_icp.predict(icp_values)])
                    all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

                # convert (B, 2, 99) into (B, 2) given width or error rate constraints
                if is_multi_ts and is_numerical:
                    significances, confs = get_ts_conf_range(all_confs,
                                                             df_target_stddev=ns.analysis['df_target_stddev'],
                                                             positive_domain=ns.positive_domain,
                                                             fixed_conf=ns.pred_args.fixed_confidence)

                    result = self._ts_assign_confs(result, X, confs, significances, ns.tss)

                elif is_numerical:
                    significances, confs = get_numeric_conf_range(all_confs,
                                                                  df_target_stddev=ns.analysis['df_target_stddev'],
                                                                  positive_domain=ns.positive_domain,
                                                                  fixed_conf=ns.pred_args.fixed_confidence)
                    result.loc[X.index, 'lower'] = confs[:, 0]
                    result.loc[X.index, 'upper'] = confs[:, 1]
                    result.loc[X.index, 'significance'] = significances

                else:
                    significances = get_categorical_conf(all_confs.squeeze())
                    result.loc[X.index, 'significance'] = significances

                # grouped time series, we replace bounds in rows that have a trained ICP
                if ns.analysis['icp'].get('__mdb_groups', False):
                    icps = ns.analysis['icp']
                    group_keys = icps['__mdb_group_keys']

                    for group in icps['__mdb_groups']:
                        icp = icps[tuple(group)]

                        # check ICP has calibration scores
                        if icp.cal_scores[0].shape[0] > 0:

                            # filter rows by group
                            X = deepcopy(icp_X)
                            for key, val in zip(group_keys, group):
                                X = X[X[key] == val]

                            if X.size > 0:
                                # set ICP caches
                                if is_multi_ts and is_numerical:
                                    target_cols = [ns.target_name] + [f'{ns.target_name}_timestep_{i}'
                                                                      for i in range(1, ns.tss.horizon)]
                                    icp.nc_function.model.prediction_cache = X[target_cols].values
                                    [X.pop(col) for col in target_cols]
                                else:
                                    icp.nc_function.model.prediction_cache = X.pop(ns.target_name).values
                                if icp.nc_function.normalizer:
                                    icp.nc_function.normalizer.prediction_cache = X.pop('__mdb_selfaware_scores').values

                                # predict and get confidence level given width or error rate constraints
                                if is_multi_ts and is_numerical:
                                    all_confs = icp.predict(X.values)
                                    fixed_conf = ns.pred_args.fixed_confidence
                                    significances, confs = get_ts_conf_range(
                                        all_confs,
                                        df_target_stddev=ns.analysis['df_target_stddev'],
                                        positive_domain=ns.positive_domain,
                                        group=tuple(group),
                                        fixed_conf=fixed_conf
                                    )
                                    result = self._ts_assign_confs(result, X, confs, significances, ns.tss)

                                elif is_numerical:
                                    all_confs = icp.predict(X.values)
                                    fixed_conf = ns.pred_args.fixed_confidence
                                    significances, confs = get_numeric_conf_range(
                                        all_confs,
                                        df_target_stddev=ns.analysis['df_target_stddev'],
                                        positive_domain=ns.positive_domain,
                                        group=tuple(group),
                                        fixed_conf=fixed_conf
                                    )

                                    # only replace where grouped ICP is more informative (i.e. tighter)
                                    if fixed_conf is None:
                                        default_widths = result.loc[X.index, 'upper'] - result.loc[X.index, 'lower']
                                        grouped_widths = np.subtract(confs[:, 1], confs[:, 0])
                                        insert_index = (default_widths > grouped_widths)[lambda x: x.isin([True])].index
                                        conf_index = (default_widths.reset_index(drop=True) >
                                                      grouped_widths)[lambda x: x.isin([True])].index

                                        result.loc[insert_index, 'lower'] = confs[conf_index, 0]
                                        result.loc[insert_index, 'upper'] = confs[conf_index, 1]
                                        result.loc[insert_index, 'significance'] = significances[conf_index]

                                else:
                                    all_ranges = np.array([icp.predict(X.values)])
                                    all_confs = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)
                                    significances = get_categorical_conf(all_confs)
                                    result.loc[X.index, 'significance'] = significances

                row_insights['confidence'] = result['significance']

                if is_numerical:
                    row_insights['lower'] = result['lower']
                    row_insights['upper'] = result['upper']

                # anomaly detection
                if is_anomaly_task:
                    row_insights['anomaly'] = None
                    if ns.target_name in ns.data.columns:
                        anomalies = get_anomalies(row_insights,
                                                  ns.data[ns.target_name],
                                                  cooldown=ns.pred_args.anomaly_cooldown)
                        if '__mdb_ts_inferred' in ns.data.columns:
                            ignore_idxs = np.where(ns.data['__mdb_ts_inferred'].values)
                            anomalies[ignore_idxs] = None
                        row_insights['anomaly'] = list(anomalies)

            if ns.tss.is_timeseries and ns.tss.horizon > 1:
                if is_numerical and ns.pred_args.simple_ts_bounds:
                    row_insights = add_tn_num_conf_bounds(row_insights, ns.tss)
                elif not is_numerical:
                    row_insights = add_tn_cat_conf_bounds(row_insights, ns.tss)

            row_insights, global_insights = self._formatted(row_insights, global_insights, ns, is_numerical)

        return row_insights, global_insights

    @staticmethod
    def _formatted(row_insights, global_insights, ns, is_numerical):
        # clip if necessary
        if is_numerical:
            lower_limit = 0.0 if ns.positive_domain else -pow(2, 62)
            upper_limit = pow(2, 62)
            if not (ns.tss.is_timeseries and ns.tss.horizon > 1):
                row_insights['upper'] = row_insights['upper'].clip(lower_limit, upper_limit)
                row_insights['lower'] = row_insights['lower'].clip(lower_limit, upper_limit)
                row_insights['prediction'] = row_insights['prediction'].clip(lower_limit, upper_limit)
            else:
                row_insights['upper'] = [np.array(row).clip(lower_limit, upper_limit).tolist()
                                         for row in row_insights['upper']]
                row_insights['lower'] = [np.array(row).clip(lower_limit, upper_limit).tolist()
                                         for row in row_insights['lower']]
                row_insights['prediction'] = [np.array(row).clip(lower_limit, upper_limit).tolist()
                                              for row in row_insights['prediction']]

        # Make sure the target and real values are of an appropriate type
        if ns.target_dtype in (dtype.integer, ):
            row_insights['prediction'] = row_insights['prediction'].astype(int)
            row_insights['upper'] = row_insights['upper'].astype(int)
            row_insights['lower'] = row_insights['lower'].astype(int)

        elif ns.target_dtype in (dtype.float, dtype.quantity):
            row_insights['prediction'] = row_insights['prediction'].astype(float)
            row_insights['upper'] = row_insights['upper'].astype(float)
            row_insights['lower'] = row_insights['lower'].astype(float)

        elif ns.target_dtype in (dtype.short_text, dtype.rich_text, dtype.binary, dtype.categorical):
            row_insights['prediction'] = row_insights['prediction'].astype(str)

        # horizon collapse
        if ns.tss.is_timeseries and is_numerical and ns.tss.horizon > 1:
            row_insights['prediction_sum'] = row_insights['prediction'].apply(lambda x: sum(x))
            row_insights['lower_sum'] = row_insights['lower'].apply(lambda x: min(x))
            row_insights['upper_sum'] = row_insights['upper'].apply(lambda x: max(x))
            row_insights['confidence_mean'] = row_insights['confidence'].apply(lambda x: np.mean(x))

        return row_insights, global_insights

    @staticmethod
    def _ts_assign_confs(result, df, confs, significances, tss) -> pd.DataFrame:
        result.loc[df.index, 'lower'] = confs[:, 0, 0]
        result.loc[df.index, 'upper'] = confs[:, 0, 1]
        result.loc[df.index, 'significance'] = significances[:, 0]
        for timestep in range(1, tss.horizon):
            result.loc[df.index, f'lower_timestep_{timestep}'] = confs[:, timestep, 0]
            result.loc[df.index, f'upper_timestep_{timestep}'] = confs[:, timestep, 1]
            result.loc[df.index,
                       f'significance_timestep_{timestep}'] = significances[:, timestep]

        # TODO: only if tighter
        # merge all significances, lower and upper bounds into a single column
        for base_col in ['significance', 'lower', 'upper']:
            added_cols = [f'{base_col}_timestep_{t}' for t in range(1, tss.horizon)]
            cols = [base_col] + added_cols
            result.loc[df.index, base_col] = result.loc[df.index, cols].values.tolist()
            # result[base_col] = result[cols].values.tolist()

        return result
