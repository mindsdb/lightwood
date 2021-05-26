import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from nc.icp import IcpRegressor, IcpClassifier
from nc.nc import RegressorNc, ClassifierNc, MarginErrFunc

from lightwood.model.nn import Nn
from mindsdb_native.libs.helpers.general_helpers import pickle_obj, evaluate_accuracy
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.helpers.conformal_helpers import ConformalClassifierAdapter, ConformalRegressorAdapter
from mindsdb_native.libs.helpers.conformal_helpers import BoostedAbsErrorErrFunc, SelfawareNormalizer
from mindsdb_native.libs.helpers.confidence_helpers import clean_df, set_conf_range
from mindsdb_native.libs.helpers.accuracy_stats import AccStats


def model_analyzer(predictor, config, data):
    np.seterr(divide='warn', invalid='warn')
    np.random.seed(0)
    """
    # Runs the model on the validation set in order to evaluate the accuracy and confidence of future predictions
    """
    train_df = data.train_df
    if predictor.lmd['tss']['is_timeseries']:
        train_df = data.train_df[data.train_df['make_predictions'] == True]

    validation_df = data.validation_df
    if predictor.lmd['tss']['is_timeseries']:
        validation_df = data.validation_df[data.validation_df['make_predictions'] == True]

    test_df = data.test_df
    if predictor.lmd['tss']['is_timeseries']:
        test_df = data.test_df[data.test_df['make_predictions'] == True]

    output_columns = predictor.lmd['predict_columns']
    input_columns = [col for col in predictor.lmd['columns'] if col not in output_columns and col not in predictor.lmd['columns_to_ignore']]

    # Make predictions on the validation and test datasets
    predictor.config['include_extra_data'] = True
    normal_predictions = predictor.predict('validate')
    normal_predictions_test = predictor.predict('test')
    predictor.lmd['test_data_plot'] = {}

    # confidence estimation with inductive conformal predictors (ICPs)
    predictor.hmd['label_encoders'] = {}   # needed to represent categorical labels inside nonconformist
    predictor.hmd['icp'] = {'__mdb_active': False}

    for target in output_columns:
        typing_info = predictor.lmd['stats_v2'][target]['typing']
        data_type = typing_info['data_type']
        data_subtype = typing_info['data_subtype']

        is_numerical = data_type == DATA_TYPES.NUMERIC or \
                       (data_type == DATA_TYPES.SEQUENTIAL and
                        DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys())

        is_classification = data_type == DATA_TYPES.CATEGORICAL or \
                            (data_type == DATA_TYPES.SEQUENTIAL and
                             DATA_TYPES.CATEGORICAL in typing_info['data_type_dist'].keys())

        is_multi_ts = predictor.lmd['tss']['is_timeseries'] and \
                      predictor.lmd['tss']['nr_predictions'] > 1

        is_selfaware = isinstance(predictor._mixer, Nn) and \
                       predictor._mixer.is_selfaware

        fit_params = {
            'columns_to_ignore': [col for col in output_columns if col != target],
            'nr_preds': predictor.lmd['tss'].get('nr_predictions', 0)
        }
        fit_params['columns_to_ignore'].extend([f'{target}_timestep_{i}' for i in range(1, fit_params['nr_preds'])])

        if is_classification:
            if data_subtype != DATA_SUBTYPES.TAGS:
                all_classes = np.array(predictor.lmd['stats_v2'][target]['histogram']['x'])
                enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
                enc.fit(all_classes.reshape(-1, 1))
                predictor.hmd['label_encoders'][target] = enc
            else:
                predictor.hmd['label_encoders'][target] = None

            adapter = ConformalClassifierAdapter
            nc_function = MarginErrFunc()
            nc_class = ClassifierNc
            icp_class = IcpClassifier

        else:
            adapter = ConformalRegressorAdapter
            nc_function = BoostedAbsErrorErrFunc()
            nc_class = RegressorNc
            icp_class = IcpRegressor

        if is_numerical or (is_classification and data_subtype != DATA_SUBTYPES.TAGS):
            model = adapter(predictor)

            if is_selfaware:
                norm_params = {'output_column': target}
                normalizer = SelfawareNormalizer(fit_params=norm_params)
                normalizer.prediction_cache = normal_predictions.get(f'{target}_selfaware_scores', None)
            else:
                normalizer = None

            fixed_significance = predictor.lmd.get('fixed_confidenece', None)

            # instance the ICP
            nc = nc_class(model, nc_function, normalizer=normalizer)
            icp = icp_class(nc)
            predictor.hmd['icp'][target] = {'__default': icp}

            # setup prediction cache to avoid additional .predict() calls
            if is_classification:
                icp.nc_function.model.prediction_cache = np.array(normal_predictions[f'{target}_class_distribution'])
                icp.nc_function.model.class_map = predictor.lmd['stats_v2'][target]['lightwood_class_map']
            elif is_multi_ts:
                # we fit ICPs for time series confidence bounds only at t+1 forecast
                icp.nc_function.model.prediction_cache = np.array([p[0] for p in normal_predictions[target]])
            else:
                icp.nc_function.model.prediction_cache = np.array(normal_predictions[target])

            # "fit" the default ICP
            predictor.hmd['icp'][target]['__default'].fit(None, None)
            if not is_classification:
                predictor.lmd['stats_v2'][target]['train_std_dev'] = {'__default': train_df[target].std()}

            # fit additional ICPs in time series tasks with grouped columns
            if predictor.lmd['tss']['is_timeseries'] and predictor.lmd['tss']['group_by']:

                # create an ICP for each possible group
                group_info = data.train_df[predictor.lmd['tss']['group_by']].to_dict('list')
                all_group_combinations = list(product(*[set(x) for x in group_info.values()]))
                predictor.hmd['icp'][target]['__mdb_groups'] = all_group_combinations
                predictor.hmd['icp'][target]['__mdb_group_keys'] = [x for x in group_info.keys()]

                for combination in all_group_combinations:
                    # frozenset lets us hash
                    predictor.hmd['icp'][target][frozenset(combination)] = deepcopy(icp)
                    predictor.hmd['icp'][target][frozenset(combination)].fit(None, None)

            # calibrate ICP
            icp_df = deepcopy(data.cached_val_df)
            icp_df, y = clean_df(icp_df, target, predictor, is_classification, fit_params)
            predictor.hmd['icp'][target]['__default'].index = icp_df.columns
            predictor.hmd['icp'][target]['__default'].calibrate(icp_df.values, y)

            # get confidence estimation for validation dataset
            _, ranges = set_conf_range(icp_df, icp, target, typing_info, predictor.lmd,
                                       significance=fixed_significance)
            if not is_classification:
                result_df = pd.DataFrame(index=data.cached_val_df.index, columns=['lower', 'upper'])
                result_df.loc[icp_df.index, 'lower'] = ranges[:, 0]
                result_df.loc[icp_df.index, 'upper'] = ranges[:, 1]

            # calibrate additional grouped ICPs
            if predictor.lmd['tss']['is_timeseries'] and predictor.lmd['tss']['group_by']:
                icps = predictor.hmd['icp'][target]
                group_keys = icps['__mdb_group_keys']

                # add all predictions to the cached DF
                icps_df = deepcopy(data.cached_val_df)
                if is_multi_ts:
                    icps_df[f'__predicted_{target}'] = [p[0] for p in normal_predictions[target]]
                else:
                    icps_df[f'__predicted_{target}'] = normal_predictions[target]

                for group in icps['__mdb_groups']:
                    icp_df = icps_df

                    if is_selfaware:
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
                        icp_train_df = train_df
                        for key, val in zip(group_keys, group):
                            icp_train_df = icp_train_df[icp_train_df[key] == val]
                        y_train = icp_train_df[target].values
                        predictor.lmd['stats_v2'][target]['train_std_dev'][frozenset(group)] = y_train.std()

                    # get bounds for relevant rows in validation dataset
                    _, group_ranges = set_conf_range(icp_df, icps[frozenset(group)], target, typing_info,
                                                     predictor.lmd, group=frozenset(group),
                                                     significance=fixed_significance)
                    # save group bounds
                    if not is_classification:
                        result_df.loc[icp_df.index, 'lower'] = group_ranges[:, 0]
                        result_df.loc[icp_df.index, 'upper'] = group_ranges[:, 1]

            # consolidate all groups here
            if not is_classification:
                ranges = result_df.values
                normal_predictions[f'{target}_confidence_range'] = ranges

            predictor.hmd['icp']['__mdb_active'] = True

    # get accuracy metric
    normal_accuracy = evaluate_accuracy(
        normal_predictions,
        validation_df,
        predictor.lmd['stats_v2'],
        output_columns,
        backend=predictor
    )

    empty_input_predictions = {}
    empty_input_accuracy = {}
    empty_input_predictions_test = {}

    if not predictor.lmd['disable_column_importance']:
        ignorable_input_columns = [x for x in input_columns if predictor.lmd['stats_v2'][x]['typing']['data_type'] != DATA_TYPES.FILE_PATH
                                   and (not predictor.lmd['tss']['is_timeseries'] or
                                        (x not in predictor.lmd['tss']['order_by'] and
                                         x not in predictor.lmd['tss']['historical_columns']))]

        for col in ignorable_input_columns:
            empty_input_predictions[col] = predictor.predict('validate', ignore_columns=[col])
            empty_input_predictions_test[col] = predictor.predict('test', ignore_columns=[col])
            empty_input_accuracy[col] = evaluate_accuracy(
                empty_input_predictions[col],
                validation_df,
                predictor.lmd['stats_v2'],
                output_columns,
                backend=predictor
            )

        # Get some information about the importance of each column
        predictor.lmd['column_importances'] = {}
        for col in ignorable_input_columns:
            accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
            # normalize from 0 to 10
            predictor.lmd['column_importances'][col] = 10 * max(0, accuracy_increase)
            assert predictor.lmd['column_importances'][col] <= 10

    # Get accuracy stats
    overall_accuracy_arr = []
    predictor.lmd['accuracy_histogram'] = {}
    predictor.lmd['confusion_matrices'] = {}
    predictor.lmd['accuracy_samples'] = {}
    predictor.hmd['acc_stats'] = {}

    predictor.lmd['train_data_accuracy'] = {}
    predictor.lmd['test_data_accuracy'] = {}
    predictor.lmd['valid_data_accuracy'] = {}

    for col in output_columns:

        # Training data accuracy
        predictions = predictor.predict('predict_on_train_data')
        predictor.lmd['train_data_accuracy'][col] = evaluate_accuracy(
            predictions,
            data.train_df,
            predictor.lmd['stats_v2'],
            [col],
            backend=predictor
        )

        # Testing data accuracy
        predictor.lmd['test_data_accuracy'][col] = evaluate_accuracy(
            normal_predictions_test,
            test_df,
            predictor.lmd['stats_v2'],
            [col],
            backend=predictor
        )

        # Validation data accuracy
        predictor.lmd['valid_data_accuracy'][col] = evaluate_accuracy(
            normal_predictions,
            validation_df,
            predictor.lmd['stats_v2'],
            [col],
            backend=predictor
        )

    for col in output_columns:
        acc_stats = AccStats(
            col_stats=predictor.lmd['stats_v2'][col],
            col_name=col,
            input_columns=input_columns
        )

        predictions_arr = [normal_predictions_test] + [x for x in empty_input_predictions_test.values()]

        acc_stats.fit(
            test_df,
            predictions_arr,
            [[ignored_column] for ignored_column in empty_input_predictions_test]
        )

        overall_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats()
        overall_accuracy_arr.append(overall_accuracy)

        predictor.lmd['accuracy_histogram'][col] = accuracy_histogram
        predictor.lmd['confusion_matrices'][col] = cm
        predictor.lmd['accuracy_samples'][col] = accuracy_samples
        predictor.hmd['acc_stats'][col] = pickle_obj(acc_stats)

    predictor.lmd['validation_set_accuracy'] = normal_accuracy
    if predictor.lmd['stats_v2'][col]['typing']['data_type'] == DATA_TYPES.NUMERIC:
        predictor.lmd['validation_set_accuracy_r2'] = normal_accuracy
