from typing import Dict, List, Optional

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
from lightwood.encoder.text.pretrained import PretrainedLangEncoder

from lightwood.analysis.acc_stats import AccStats
from lightwood.analysis.nc.norm import Normalizer
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
    train_data: List[EncodedDs],
    stats_info: StatisticalAnalysis,
    target: str,
    ts_cfg: TimeseriesSettings,
    dtype_dict: Dict[str, str],
    disable_column_importance: bool,
    fixed_significance: float,
    positive_domain: bool,
    confidence_normalizer: bool,
    accuracy_functions,
    analysis_blocks: Optional = []
):
    """Analyses model on a validation fold to evaluate accuracy and confidence of future predictions"""

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
    is_classification = data_type in (dtype.categorical, dtype.binary)
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    # encoded data representations
    encoded_train_data = ConcatedEncodedDs(train_data)
    encoded_val_data = ConcatedEncodedDs(data)
    data = encoded_val_data.data_frame

    # additional flags
    has_pretrained_text_enc = any([isinstance(enc, PretrainedLangEncoder)
                                   for enc in encoded_train_data.encoders.values()])
    disable_column_importance = disable_column_importance or ts_cfg.is_timeseries or has_pretrained_text_enc

    input_cols = list([col for col in data.columns if col != target])
    normal_predictions = predictor(encoded_val_data) if not is_classification else predictor(
        encoded_val_data, predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    # core analysis methods:
    # 1. confidence estimation with inductive conformal predictors (ICPs)
    icp_output, result_df = icp_calibration(
        predictor,
        target,
        dtype_dict,
        normal_predictions,
        data,
        train_data,
        encoded_val_data,
        is_classification,
        is_numerical,
        is_multi_ts,
        stats_info,
        ts_cfg,
        fixed_significance,
        positive_domain,
        confidence_normalizer,
    )
    runtime_analyzer = {**runtime_analyzer, **icp_output}

    # 2. accuracy metric for validation data
    score_dict = evaluate_accuracy(data, normal_predictions['prediction'], target, accuracy_functions)
    normal_accuracy = np.mean(list(score_dict.values()))

    # 3. global feature importance
    if not disable_column_importance:
        column_importances = compute_global_importance(
            predictor,
            input_cols,
            target,
            data,
            encoded_val_data,
            normal_accuracy,
            accuracy_functions,
            is_classification,
            ts_cfg
        )
    else:
        column_importances = None

    # 4. validation stats (e.g. confusion matrix, histograms)
    acc_stats = AccStats(dtype_dict=dtype_dict, target=target, buckets=stats_info.buckets)
    acc_stats.fit(data, normal_predictions, conf=result_df)
    bucket_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats(
        is_classification=is_classification, is_numerical=is_numerical)
    runtime_analyzer['bucket_accuracy'] = bucket_accuracy

    model_analysis = ModelAnalysis(
        accuracies=score_dict,
        accuracy_histogram=accuracy_histogram,
        accuracy_samples=accuracy_samples,
        train_sample_size=len(encoded_train_data),
        test_sample_size=len(encoded_val_data),
        confusion_matrix=cm,
        column_importances=column_importances,
        histograms=stats_info.histograms,
        dtypes=dtype_dict
    )

    # user analysis blocks
    for block in analysis_blocks:
        runtime_analyzer = block.compute(runtime_analyzer, **{})

    return model_analysis, runtime_analyzer


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

        adapter = ConformalClassifierAdapter
        nc_function = MarginErrFunc()
        nc_class = ClassifierNc
        icp_class = IcpClassifier

    else:
        adapter = ConformalRegressorAdapter
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


def compute_global_importance(
        predictor: BaseEnsemble,
        input_cols,
        target: str,
        val_data,
        encoded_data,
        normal_accuracy: float,
        accuracy_functions: List,
        is_classification: bool,
        ts_cfg: TimeseriesSettings
) -> dict:

    empty_input_accuracy = {}
    ignorable_input_cols = [x for x in input_cols if (not ts_cfg.is_timeseries or
                                                      (x not in ts_cfg.order_by and
                                                       x not in ts_cfg.historical_columns))]
    for col in ignorable_input_cols:
        partial_data = deepcopy(encoded_data)
        partial_data.clear_cache()
        for ds in partial_data.encoded_ds_arr:
            ds.data_frame[col] = [None] * len(ds.data_frame[col])

        if not is_classification:
            empty_input_preds = predictor(partial_data)
        else:
            empty_input_preds = predictor(partial_data, predict_proba=True)

        empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
            val_data,
            empty_input_preds['prediction'],
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

    return column_importances
