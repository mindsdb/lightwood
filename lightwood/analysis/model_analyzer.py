from typing import Dict, List

from torch.utils.data.dataset import ConcatDataset
from lightwood.api.types import Feature, ModelAnalysis, Output, ProblemDefinition, StatisticalAnalysis, TimeseriesSettings
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import OneHotEncoder

from lightwood.ensemble import BaseEnsemble
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.api import dtype
from lightwood.helpers.general import evaluate_accuracy

from lightwood.analysis.acc_stats import AccStats
from lightwood.analysis.nc.norm import SelfawareNormalizer
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc
from lightwood.analysis.nc.util import clean_df, set_conf_range
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.nc import RegressorNc, ClassifierNc, MarginErrFunc
from lightwood.analysis.nc.wrappers import ConformalClassifierAdapter, ConformalRegressorAdapter


"""
[31/5/21] Analyzer roadmap:

- flow works for categorical and numerical with minimal adaptation to code logic, should include accStats, 
      global feat importance and ICP confidence
     - [DONE, 3/6/21] icp confidence
     - [] use class distribution output
     - [] global feat importance
     - [partially done] accStats
- streamline nonconformist custom implementation to cater analysis needs
- introduce model-agnostic normalizer (previously known as self aware NN)
- re-introduce time series (and grouped ICPs)
    - [2/7/2021] currently working on this
"""


def model_analyzer(
        predictor: BaseEnsemble,
        data: List[EncodedDs],
        stats_info: StatisticalAnalysis,
        target: str,
        ts_cfg: TimeseriesSettings,
        dtype_dict: Dict[str, str],
        disable_column_importance,
        fixed_significance,
        positive_domain,
        accuracy_functions
    ):
    """Analyses model on a validation fold to evaluate accuracy and confidence of future predictions"""

    # @ TODO: reimplement time series
    # validation_df = data.validation_df
    # if ts_cfg.is_timeseries:
    #     validation_df = data.validation_df[data.validation_df['__mdb_make_predictions'] == True]
    # ... same with test and train dfs
    encoded_data = ConcatedEncodedDs(data)
    data = encoded_data.data_frame
    runtime_analyzer = {}
    predictions = {}
    input_cols = list([col for col in data.columns if col != target])
    normal_predictions = predictor(encoded_data)  # TODO: this should include beliefs for categorical targets
    normal_predictions = normal_predictions.set_index(data.index)

    # confidence estimation with inductive conformal predictors (ICPs)
    runtime_analyzer['icp'] = {'__mdb_active': False}

    # typing_info = stats_info['typing']
    data_type = dtype_dict[target]
    data_subtype = data_type

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
                   # and dtype.numeric in typing_info['data_type_dist'].keys())

    is_classification = data_type in (dtype.categorical, dtype.binary)
                        # dtype.categorical in typing_info['data_type_dist'].keys())

    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    fit_params = {
        'nr_preds': ts_cfg.nr_predictions or 0,
        'columns_to_ignore': []
    }
    fit_params['columns_to_ignore'].extend([f'timestep_{i}' for i in range(1, fit_params['nr_preds'])])

    # @TODO: adapters should not be needed anymore
    if is_classification:
        if data_subtype != dtype.tags:
            all_classes = np.array(stats_info.train_observed_classes)
            enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
            enc.fit(all_classes.reshape(-1, 1))
            runtime_analyzer['label_encoders'] = enc  # needed to repr cat labels inside nonconformist @TODO: remove?
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

        norm_params = {'output_column': target}
        normalizer = SelfawareNormalizer(fit_params=norm_params)
        normalizer.prediction_cache = normal_predictions.get('selfaware_scores', None)  # @TODO: call to .explain()

        # instance the ICP
        nc = nc_class(model, nc_function)  # , normalizer=normalizer)  # @TODO: reintroduce normalizer
        icp = icp_class(nc)

        runtime_analyzer['icp']['__default'] = icp  # @TODO: better typing for output

        # setup prediction cache to avoid additional .predict() calls
        if is_classification:
            if False:  # config.output.returns_proba:
                # @TODO: models should indicate whether they predict prob beliefs. if so, use them here
                icp.nc_function.model.prediction_cache = np.array(normal_predictions['class_distribution'])
                icp.nc_function.model.class_map = stats_info['lightwood_class_map']
            else:
                class_map = {i: v for i, v in enumerate(stats_info.train_observed_classes)}
                predicted_classes = pd.get_dummies(normal_predictions['prediction']).values  # inflate to one-hot enc

                icp.nc_function.model.prediction_cache = predicted_classes
                icp.nc_function.model.class_map = class_map  # @TODO: still needed?
        elif is_multi_ts:
            # we fit ICPs for time series confidence bounds only at t+1 forecast
            icp.nc_function.model.prediction_cache = np.array([p[0] for p in normal_predictions['prediction']])
        else:
            icp.nc_function.model.prediction_cache = np.array(normal_predictions['prediction'])

        runtime_analyzer['icp']['__default'].fit(None, None)  # @TODO: rm fit call after v1 works, 'twas a hack from the start
        if not is_classification:
            runtime_analyzer['train_std_dev'] = {'__default': stats_info.train_std_dev}

        # fit additional ICPs in time series tasks with grouped columns
        if ts_cfg.is_timeseries and ts_cfg.group_by:

            # create an ICP for each possible group
            group_info = data[ts_cfg.group_by].to_dict('list')  # @TODO: should save this info from all data in timeseries_analyzer then send it here. Fow now, validation only means it can have incomplete data
            all_group_combinations = list(product(*[set(x) for x in group_info.values()]))
            runtime_analyzer['icp']['__mdb_groups'] = all_group_combinations
            runtime_analyzer['icp']['__mdb_group_keys'] = [x for x in group_info.keys()]

            for combination in all_group_combinations:
                # frozenset lets us hash
                runtime_analyzer['icp'][frozenset(combination)] = deepcopy(icp)
                runtime_analyzer['icp'][frozenset(combination)].fit(None, None)

        # calibrate ICP
        icp_df = deepcopy(data)
        icp_df, y = clean_df(icp_df, target, is_classification, runtime_analyzer.get('label_encoders', None))
        runtime_analyzer['icp']['__default'].index = icp_df.columns
        runtime_analyzer['icp']['__default'].calibrate(icp_df.values, y)

        # get confidence estimation for validation dataset
        conf, ranges = set_conf_range(icp_df, icp, dtype_dict[target], runtime_analyzer, positive_domain=positive_domain, significance=fixed_significance)
        if not is_classification:
            # @TODO previously using cached_val_df index, analyze how to replicate once again for the TS case
            # @TODO once using normalizer, add column for confidence proper here, and return DF in categorical case too
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

            # add all predictions to the cached DF
            icps_df = deepcopy(data)  # @TODO: previously used cached_val_df
            if is_multi_ts:
                icps_df[f'__predicted_{target}'] = [p[0] for p in normal_predictions['prediction']]
            else:
                icps_df[f'__predicted_{target}'] = normal_predictions['prediction']

            for group in icps['__mdb_groups']:
                icp_df = icps_df
                if icps[frozenset(group)].nc_function.normalizer is not None:  # @TODO: reintroduce normalizer
                    icp_df[f'__selfaware_{target}'] = icps[frozenset(group)].nc_function.normalizer.prediction_cache

                # filter irrelevant rows for each group combination
                for key, val in zip(group_keys, group):
                    icp_df = icp_df[icp_df[key] == val]

                # save relevant predictions in the caches, then calibrate the ICP
                pred_cache = icp_df.pop(f'__predicted_{target}').values
                icps[frozenset(group)].nc_function.model.prediction_cache = pred_cache
                icp_df, y = clean_df(icp_df, target, is_classification, runtime_analyzer.get('label_encoders', None))
                if icps[frozenset(group)].nc_function.normalizer is not None:  # @TODO: reintroduce normalizer
                    icps[frozenset(group)].nc_function.normalizer.prediction_cache = icp_df.pop(f'__selfaware_{target}').values

                icps[frozenset(group)].index = icp_df.columns      # important at inference time
                icps[frozenset(group)].calibrate(icp_df.values, y)

                # save training std() for bounds width selection
                if not is_classification:
                    icp_train_df = data
                    for key, val in zip(group_keys, group):
                        icp_train_df = icp_train_df[icp_train_df[key] == val]
                    y_train = icp_train_df[target].values
                    runtime_analyzer['train_std_dev'][frozenset(group)] = y_train.std()  # @TODO: check that this is indeed train std dev

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

        # TODO: should we pass observed confidences in validation dataset?
        runtime_analyzer['icp']['__mdb_active'] = True

    # TODO: calculate acc on other folds?
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

    # @TODO: reactivate global feature importance
    if not disable_column_importance:
        ignorable_input_cols = [x for x in input_cols if (not ts_cfg.is_timeseries or
                                                                (x not in ts_cfg.order_by and
                                                                 x not in ts_cfg.historical_columns))]
        for col in ignorable_input_cols:
            empty_input_predictions[col] = predictor('validate', ignore_columns=[col])  # @TODO: add this param?
            empty_input_accuracy[col] = np.mean(list(evaluate_accuracy(
                data,
                empty_input_predictions[col]
            ).values()))

        # Get some information about the importance of each column
        # @TODO: Figure out if it's too slow
        column_importances = {}
        for col in ignorable_input_cols:
            accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
            # normalize from 0 to 10
            column_importances[col] = 10 * max(0, accuracy_increase)
    else:
        column_importances = None

    # @TODO: Training / testing data accuracy here ?

    acc_stats = AccStats(dtype_dict=dtype_dict, target=target)

    predictions_arr = [normal_predictions['prediction'].values.flatten().tolist()] + [x for x in empty_input_predictions_test.values()]

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