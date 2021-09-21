from typing import Dict
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from lightwood.api.dtype import dtype
from lightwood.ensemble.base import BaseEnsemble
from lightwood.data.encoded_ds import ConcatedEncodedDs
from lightwood.api.types import StatisticalAnalysis, TimeseriesSettings

from lightwood.analysis.nc.norm import Normalizer
from lightwood.analysis.nc.util import clean_df, set_conf_range
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.base import CachedRegressorAdapter, CachedClassifierAdapter
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc, RegressorNc, ClassifierNc, MarginErrFunc


"""
Pending:
 - [] simplify nonconformist custom implementation
 - [] reimplement caching for faster analysis?
 - [] confidence for T+N <- active research question
"""


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
