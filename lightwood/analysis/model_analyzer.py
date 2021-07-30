from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import OneHotEncoder

from lightwood.api import dtype
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.helpers.general import evaluate_accuracy
from lightwood.ensemble import BaseEnsemble

from lightwood.analysis.acc_stats import AccStats
from lightwood.analysis.nc.norm import SelfawareNormalizer
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc
from lightwood.analysis.nc.util import clean_df, set_conf_range
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.nc import RegressorNc, ClassifierNc, MarginErrFunc
from lightwood.analysis.nc.wrappers import ConformalClassifierAdapter, ConformalRegressorAdapter, t_softmax


"""
Pending:
 - [] simplify nonconformist custom implementation to deprecate wrappers
 - [] reimplement caching for faster analysis?
 - [] confidence for T+N <- active research question
"""


def model_analyzer(
        predictor: BaseEnsemble,
        data: List[EncodedDs],
        encoded_train_data: ConcatedEncodedDs,
        stats_info: StatisticalAnalysis,
        target: str,
        ts_cfg: TimeseriesSettings,
        dtype_dict: Dict[str, str],
        disable_column_importance: bool,
        fixed_significance: float,
        positive_domain: bool,
        accuracy_functions
    ):
    """Analyses model on a validation fold to evaluate accuracy and confidence of future predictions"""

    data_type = dtype_dict[target]
    data_subtype = data_type

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
    is_classification = data_type in (dtype.categorical, dtype.binary)
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    encoded_data = ConcatedEncodedDs(data)
    data = encoded_data.data_frame
    runtime_analyzer = {}
    predictions = {}
    input_cols = list([col for col in data.columns if col != target])
    normal_predictions = predictor(encoded_data) if not is_classification else predictor(encoded_data, predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    # confidence estimation with inductive conformal predictors (ICPs)
    runtime_analyzer['icp'] = {'__mdb_active': False}

    all_cat_cols = [col for col in normal_predictions.columns if '__mdb_proba' in col]
    all_classes = np.array([col.replace('__mdb_proba_', '') for col in all_cat_cols])
    fit_params = {'nr_preds': ts_cfg.nr_predictions or 0, 'columns_to_ignore': [] }
    fit_params['columns_to_ignore'].extend([f'timestep_{i}' for i in range(1, fit_params['nr_preds'])])

    if is_classification:
        if data_subtype != dtype.tags:
            enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
            enc.fit(all_classes.reshape(-1, 1))
            runtime_analyzer['label_encoders'] = enc  # needed to repr cat labels inside nonconformist
        else:
            runtime_analyzer['label_encoders'] = None

        adapter = ConformalClassifierAdapter
        nc_function = MarginErrFunc()
        nc_class = ClassifierNc
        icp_class = IcpClassifier

    else:
        adapter = ConformalRegressorAdapter
        nc_function = BoostedAbsErrorErrFunc()
        nc_class = RegressorNc
        icp_class = IcpRegressor

    if is_numerical or (is_classification and data_subtype != dtype.tags):
        model = adapter(predictor)

        norm_params = {'target': target, 'dtype_dict': dtype_dict, 'predictor': predictor, 'encoders': encoded_data.encoders, 'is_multi_ts': is_multi_ts}
        normalizer = SelfawareNormalizer(fit_params=norm_params)
        normalizer.fit(encoded_train_data, target)
        normalizer.prediction_cache = normalizer.predict(encoded_data)

        # instance the ICP
        nc = nc_class(model, nc_function, normalizer=normalizer)
        icp = icp_class(nc)

        runtime_analyzer['icp']['__default'] = icp

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
            runtime_analyzer['df_std_dev'] = {'__default': stats_info.df_std_dev}

        # fit additional ICPs in time series tasks with grouped columns
        if ts_cfg.is_timeseries and ts_cfg.group_by:

            # create an ICP for each possible group
            group_info = data[ts_cfg.group_by].to_dict('list')
            all_group_combinations = list(product(*[set(x) for x in group_info.values()]))
            runtime_analyzer['icp']['__mdb_groups'] = all_group_combinations
            runtime_analyzer['icp']['__mdb_group_keys'] = [x for x in group_info.keys()]

            for combination in all_group_combinations:
                runtime_analyzer['icp'][frozenset(combination)] = deepcopy(icp)

        # calibrate ICP
        icp_df = deepcopy(data)
        icp_df, y = clean_df(icp_df, target, is_classification, runtime_analyzer.get('label_encoders', None))
        runtime_analyzer['icp']['__default'].index = icp_df.columns
        runtime_analyzer['icp']['__default'].calibrate(icp_df.values, y)

        # get confidence estimation for validation dataset
        conf, ranges = set_conf_range(icp_df, icp, dtype_dict[target], runtime_analyzer, positive_domain=positive_domain, significance=fixed_significance)
        if not is_classification:
            result_df = pd.DataFrame(index=data.index, columns=['confidence', 'lower', 'upper'], dtype=float)
            result_df.loc[icp_df.index, 'lower'] = ranges[:, 0]
            result_df.loc[icp_df.index, 'upper'] = ranges[:, 1]
        else:
            result_df = pd.DataFrame(index=data.index, columns=['confidence'], dtype=float)

        result_df.loc[icp_df.index, 'confidence'] = conf

        # calibrate additional grouped ICPs
        if ts_cfg.is_timeseries and ts_cfg.group_by:
            icps = runtime_analyzer['icp']
            group_keys = icps['__mdb_group_keys']

            # add all predictions to DF
            icps_df = deepcopy(data)
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
                icp_df, y = clean_df(icp_df, target, is_classification, runtime_analyzer.get('label_encoders', None))
                if icps[frozenset(group)].nc_function.normalizer is not None:
                    icps[frozenset(group)].nc_function.normalizer.prediction_cache = icp_df.pop(f'__norm_{target}').values

                icps[frozenset(group)].index = icp_df.columns      # important at inference time
                icps[frozenset(group)].calibrate(icp_df.values, y)

                # save training std() for bounds width selection
                if not is_classification:
                    icp_train_df = data
                    for key, val in zip(group_keys, group):
                        icp_train_df = icp_train_df[icp_train_df[key] == val]
                    y_train = icp_train_df[target].values
                    runtime_analyzer['df_std_dev'][frozenset(group)] = y_train.std()

                # get bounds for relevant rows in validation dataset
                conf, group_ranges = set_conf_range(icp_df, icps[frozenset(group)], dtype_dict[target], runtime_analyzer,
                                                    group=frozenset(group),
                                                    positive_domain=positive_domain,
                                                    significance=fixed_significance)
                # save group bounds
                if not is_classification:
                    result_df.loc[icp_df.index, 'lower'] = group_ranges[:, 0]
                    result_df.loc[icp_df.index, 'upper'] = group_ranges[:, 1]

                result_df.loc[icp_df.index, 'confidence'] = conf

        # consolidate all groups here
        if not is_classification:
            ranges = result_df.values
            predictions['confidence_range'] = ranges

        runtime_analyzer['icp']['__mdb_active'] = True

    # get accuracy metric for validation data
    score_dict = evaluate_accuracy(
        data,
        normal_predictions['prediction'],
        target,
        accuracy_functions
    )
    normal_accuracy = np.mean(list(score_dict.values()))

    empty_input_predictions = {}
    empty_input_accuracy = {}
    empty_input_predictions_test = {}

    # compute global feature importance
    if not disable_column_importance:
        ignorable_input_cols = [x for x in input_cols if (not ts_cfg.is_timeseries or
                                                                (x not in ts_cfg.order_by and
                                                                 x not in ts_cfg.historical_columns))]
        for col in ignorable_input_cols:
            partial_data = deepcopy(encoded_data)
            partial_data.clear_cache()
            for ds in partial_data.encoded_ds_arr:
                ds.data_frame[col].values[:] = 0

            if not is_classification:
                empty_input_predictions[col] = predictor(partial_data)
            else:
                empty_input_predictions[col] = predictor(partial_data, predict_proba=True)

            empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
                data,
                empty_input_predictions[col]['prediction'],
                target,
                accuracy_functions
            ).values()))

        column_importances = {}
        acc_increases = []
        for col in ignorable_input_cols:
            accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
            acc_increases.append(accuracy_increase)

        # low 0.2 temperature to accentuate differences
        acc_increases = t_softmax(torch.Tensor([acc_increases]), t=0.2).tolist()[0]
        for col, inc in zip(ignorable_input_cols, acc_increases):
            column_importances[col] = 10 * inc  # scores go from 0 to 10 in GUI
    else:
        column_importances = None

    predictions_arr = [normal_predictions['prediction'].values.flatten().tolist()] + \
                      [x for x in empty_input_predictions_test.values()]

    acc_stats = AccStats(dtype_dict=dtype_dict, target=target)
    acc_stats.fit(
        data,
        predictions_arr,
        [[ignored_column] for ignored_column in empty_input_predictions_test]
    )

    overall_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats()

    runtime_analyzer['overall_accuracy'] = overall_accuracy
    runtime_analyzer['accuracy_histogram'] = accuracy_histogram
    runtime_analyzer['confusion_matrices'] = cm
    runtime_analyzer['accuracy_samples'] = accuracy_samples

    runtime_analyzer['validation_set_accuracy'] = normal_accuracy
    if target in [dtype.integer, dtype.float]:
        runtime_analyzer['validation_set_accuracy_r2'] = normal_accuracy

    # TODO Properly set train_sample_size and test_sample_size
    model_analysis = ModelAnalysis(
        accuracies=score_dict,
        train_sample_size=0,
        test_sample_size=0,
        confusion_matrix=cm,
        column_importances=column_importances
    )

    return model_analysis, runtime_analyzer