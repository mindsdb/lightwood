import numpy as np
import pandas as pd
from typing import Mapping, Tuple, NamedTuple, Dict
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import OneHotEncoder

from lightwood.api import LightwoodConfig, dtype
from lightwood.model.nn import Nn
from lightwood.ensemble import BaseEnsemble
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.general import evaluate_accuracy


from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.nc import RegressorNc, ClassifierNc, MarginErrFunc
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc
from lightwood.analysis.nc.norm import SelfawareNormalizer
from lightwood.analysis.nc.util import clean_df, set_conf_range
from lightwood.analysis.nc.wrappers import ConformalClassifierAdapter, ConformalRegressorAdapter

# from mindsdb_native.libs.helpers.accuracy_stats import AccStats  # @TODO: find replacement


"""
[31/5/21] Analyzer roadmap:

- v0: flow works for categorical and numerical with minimal adaptation to code logic
- v1: streamline nonconformist custom implementation to cater analysis needs
- v2: introduce model-agnostic normalizer (previously known as self aware NN)
- v3: re-introduce time series (and grouped ICPs)

"""


def model_analyzer(
        predictor: BaseEnsemble,
        encoded_data: EncodedDs,
        data: pd.DataFrame,      # @TODO: turn data and encoded into data: Tuple(Tuple(pd.DataFrame, EncodedDs)) ?
        config: LightwoodConfig,
        disable_column_importance=False
    ):
    """Analyses model on a validation fold to evaluate accuracy and confidence of future predictions"""

    # train_df = data.train_df
    # if ts_cfg.is_timeseries:
    #     train_df = data.train_df[data.train_df['make_predictions'] == True]
    #
    # validation_df = data.validation_df
    # if ts_cfg.is_timeseries:
    #     validation_df = data.validation_df[data.validation_df['make_predictions'] == True]
    #
    # test_df = data.test_df
    # if ts_cfg.is_timeseries:
    #     test_df = data.test_df[data.test_df['make_predictions'] == True]

    print("hi")

    analysis = {}
    predictions = {}
    params = config.problem_definition
    input_columns = list(config.features.keys())
    target = config.output
    normal_predictions = predictor(encoded_data)

    # @TODO: need stats_info to be input from... type deduction phase?
    # Needs:
    #   - target histogram and std()
    #   - all target classes
    #
    #
    # Obsolete:
    #   - lightwood class map -> not needed anymore
    # stats_info = predictor.lmd['stats_v2']
    stats_info = config.statistical_analyzer

    # confidence estimation with inductive conformal predictors (ICPs)
    analysis['icp'] = {'__mdb_active': False}

    # typing_info = stats_info['typing']
    data_type = target.data_dtype
    data_subtype = data_type

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
                   # and dtype.numeric in typing_info['data_type_dist'].keys())

    is_classification = data_type in [dtype.categorical] or data_type in [dtype.array]
                        # dtype.categorical in typing_info['data_type_dist'].keys())

    ts_cfg = config.problem_definition.timeseries_settings
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    fit_params = {
        'nr_preds': ts_cfg.nr_predictions or 0,
        'columns_to_ignore': []
    }
    fit_params['columns_to_ignore'].extend([f'{target}_timestep_{i}' for i in range(1, fit_params['nr_preds'])])

    # @TODO: adapters should not be needed anymore

    if is_classification:
        if data_subtype != dtype.tags:
            all_classes = np.array(stats_info.train_observed_classes)
            enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
            enc.fit(all_classes.reshape(-1, 1))
            analysis['label_encoders'] = enc  # needed to repr cat labels inside nonconformist @TODO: remove?
        else:
            analysis['label_encoders'] = None

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
        normalizer.prediction_cache = normal_predictions.get(f'{target}_selfaware_scores', None)

        fixed_significance = params.fixed_confidence

        # instance the ICP
        nc = nc_class(model, nc_function)  # , normalizer=normalizer)
        icp = icp_class(nc)

        # noinspection PyTypeChecker
        # for now
        analysis['icp']['__default'] = icp

        # setup prediction cache to avoid additional .predict() calls
        if is_classification:
            if False:  # config.output.returns_proba:
                # @TODO: models should indicate whether they predict prob beliefs. if so, use them here
                icp.nc_function.model.prediction_cache = np.array(normal_predictions[f'{target}_class_distribution'])
                icp.nc_function.model.class_map = stats_info['lightwood_class_map']
            else:
                class_map = {i: v for i, v in enumerate(stats_info.train_observed_classes)}

                # inv_class_map = {v: i for i, v in enumerate(stats_info.train_observed_classes)}
                # [[inv_class_map[p[0]]] for p in normal_predictions.values]

                predicted_classes = pd.get_dummies(normal_predictions['predictions']).values

                icp.nc_function.model.prediction_cache = predicted_classes
                icp.nc_function.model.class_map = class_map  # @TODO: still needed?
        elif is_multi_ts:
            # we fit ICPs for time series confidence bounds only at t+1 forecast
            icp.nc_function.model.prediction_cache = np.array([p[0] for p in normal_predictions[target]])
        else:
            icp.nc_function.model.prediction_cache = np.array(normal_predictions[target])

        # "fit" the default ICP
        analysis['icp']['__default'].fit(None, None)  # @TODO: remove this step after v1 works
        if not is_classification:
            analysis['stats_v2']['train_std_dev'] = {'__default': stats_info.train_std_dev}

        # fit additional ICPs in time series tasks with grouped columns
        if ts_cfg.is_timeseries and ts_cfg.group_by:

            # create an ICP for each possible group
            group_info = data.train_df[ts_cfg.group_by].to_dict('list')
            all_group_combinations = list(product(*[set(x) for x in group_info.values()]))
            analysis['icp']['__mdb_groups'] = all_group_combinations
            analysis['icp']['__mdb_group_keys'] = [x for x in group_info.keys()]

            for combination in all_group_combinations:
                # frozenset lets us hash
                analysis['icp'][frozenset(combination)] = deepcopy(icp)
                analysis['icp'][frozenset(combination)].fit(None, None)

        # calibrate ICP
        icp_df = deepcopy(data)
        icp_df, y = clean_df(icp_df, target.name, is_classification, analysis.get('label_encoders', None))
        analysis['icp']['__default'].index = icp_df.columns
        analysis['icp']['__default'].calibrate(icp_df.values, y)

        # get confidence estimation for validation dataset
        _, ranges = set_conf_range(icp_df, icp, target, stats_info, params, significance=fixed_significance)
        if not is_classification:
            result_df = pd.DataFrame(index=data.cached_val_df.index, columns=['lower', 'upper'])
            result_df.loc[icp_df.index, 'lower'] = ranges[:, 0]
            result_df.loc[icp_df.index, 'upper'] = ranges[:, 1]

        # calibrate additional grouped ICPs
        if ts_cfg.is_timeseries and ts_cfg.group_by:
            icps = analysis['icp']
            group_keys = icps['__mdb_group_keys']

            # add all predictions to the cached DF
            icps_df = deepcopy(data.cached_val_df)
            if is_multi_ts:
                icps_df[f'__predicted_{target}'] = [p[0] for p in normal_predictions[target]]
            else:
                icps_df[f'__predicted_{target}'] = normal_predictions[target]

            for group in icps['__mdb_groups']:
                icp_df = icps_df
                icp_df[f'__selfaware_{target}'] = icps[frozenset(group)].nc_function.normalizer.prediction_cache

                # filter irrelevant rows for each group combination
                for key, val in zip(group_keys, group):
                    icp_df = icp_df[icp_df[key] == val]

                # save relevant predictions in the caches, then calibrate the ICP
                pred_cache = icp_df.pop(f'__predicted_{target}').values
                icps[frozenset(group)].nc_function.model.prediction_cache = pred_cache
                icp_df, y = clean_df(icp_df, target, predictor, is_classification, fit_params)
                if icps[frozenset(group)].nc_function.normalizer is not None:
                    icps[frozenset(group)].nc_function.normalizer.prediction_cache = icp_df.pop(f'__selfaware_{target}').values

                icps[frozenset(group)].index = icp_df.columns      # important at inference time
                icps[frozenset(group)].calibrate(icp_df.values, y)

                # save training std() for bounds width selection
                if not is_classification:
                    icp_train_df = data
                    for key, val in zip(group_keys, group):
                        icp_train_df = icp_train_df[icp_train_df[key] == val]
                    y_train = icp_train_df[target].values
                    analysis['stats_v2']['train_std_dev'][frozenset(group)] = y_train.std()

                # get bounds for relevant rows in validation dataset
                _, group_ranges = set_conf_range(icp_df, icps[frozenset(group)], target, stats_info,
                                                 params, group=frozenset(group),
                                                 significance=fixed_significance)
                # save group bounds
                if not is_classification:
                    result_df.loc[icp_df.index, 'lower'] = group_ranges[:, 0]
                    result_df.loc[icp_df.index, 'upper'] = group_ranges[:, 1]

        # consolidate all groups here
        if not is_classification:
            ranges = result_df.values
            predictions[f'{target}_confidence_range'] = ranges

        # TODO: should we pass observed confidences in validation dataset?

        analysis['icp']['__mdb_active'] = True

    # get accuracy metric
    normal_accuracy = evaluate_accuracy(
        normal_predictions,
        data,
        stats_info,
        [target.name],
        backend=predictor
    )

    empty_input_predictions = {}
    empty_input_accuracy = {}
    empty_input_predictions_test = {}

    if not disable_column_importance:
        ignorable_input_columns = [x for x in input_columns if stats_info[x]['typing']['data_type'] != dcat.file_path
                                   and (not ts_cfg.is_timeseries or
                                        (x not in ts_cfg.order_by and
                                         x not in ts_cfg.historical_columns))]

        for col in ignorable_input_columns:
            empty_input_predictions[col] = predictor.predict('validate', ignore_columns=[col])
            empty_input_predictions_test[col] = predictor.predict('test', ignore_columns=[col])
            empty_input_accuracy[col] = evaluate_accuracy(
                empty_input_predictions[col],
                data,
                stats_info,
                [target.name],
                backend=predictor
            )

        # Get some information about the importance of each column
        analysis['column_importances'] = {}
        for col in ignorable_input_columns:
            accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
            # normalize from 0 to 10
            analysis['column_importances'][col] = 10 * max(0, accuracy_increase)
            assert analysis['column_importances'][col] <= 10

    # Get accuracy stats
    overall_accuracy_arr = []
    analysis['accuracy_histogram'] = {}
    analysis['confusion_matrices'] = {}
    analysis['accuracy_samples'] = {}
    analysis['acc_stats'] = {}

    analysis['train_data_accuracy'] = {}
    analysis['test_data_accuracy'] = {}
    analysis['valid_data_accuracy'] = {}



    # Training data accuracy
    # predictions = predictor.predict('predict_on_train_data')
    # analysis['train_data_accuracy'][col] = evaluate_accuracy(
    #     predictions,
    #     data.train_df,
    #     stats_info,
    #     [col],
    #     backend=predictor
    # )
    #
    # # Testing data accuracy
    # analysis['test_data_accuracy'][col] = evaluate_accuracy(
    #     normal_predictions_test,
    #     test_df,
    #     stats_info,
    #     [col],
    #     backend=predictor
    # )

    # Validation data accuracy
    analysis['valid_data_accuracy'][target.name] = evaluate_accuracy(
        normal_predictions,
        data,
        stats_info,
        [target.name],
        backend=predictor
    )

    acc_stats = AccStats(
        col_stats=stats_info[target.name],
        col_name=col,
        input_columns=input_columns
    )

    predictions_arr = [normal_predictions] + [x for x in empty_input_predictions_test.values()]

    acc_stats.fit(
        data,
        predictions_arr,
        [[ignored_column] for ignored_column in empty_input_predictions_test]
    )

    overall_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats()
    overall_accuracy_arr.append(overall_accuracy)

    analysis['accuracy_histogram'][col] = accuracy_histogram
    analysis['confusion_matrices'][col] = cm
    analysis['accuracy_samples'][col] = accuracy_samples
    # analysis['acc_stats'][col] = pickle_obj(acc_stats) # TODO: replace pickle_obj with some saving logic

    analysis['validation_set_accuracy'] = normal_accuracy
    if stats_info[col]['typing']['data_type'] == dcat.numeric:
        analysis['validation_set_accuracy_r2'] = normal_accuracy

    return analysis, predictions