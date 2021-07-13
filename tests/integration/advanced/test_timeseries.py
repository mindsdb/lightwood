import unittest
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, balanced_accuracy_score

from lightwood.api.types import ProblemDefinition


class TestTimeseries(unittest.TestCase):
    def test_timeseries_regression(self):
        """
        Tests a regression dataset and unsupervised anomaly detection
        """
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/sunspots.csv')
        target = 'Sunspots'

        predictor = predictor_from_problem(datasource.df, ProblemDefinition.from_dict(
            {
                'target': target,
                'time_aim': 100,
                'anomaly_detection': False,
                'timeseries_settings': {
                'order_by': ['Month'],
                'use_previous_target': True,
                'window': 5
            },
        }))

        predictor.learn(datasource.df)

        predictions = predictor.predict(datasource.df)

        # @TODO: Remove later and add asserts for both acc and anomalies
        print(r2_score(datasource.df[target], predictions['prediction']))
        print(mean_absolute_error(datasource.df[target], predictions['prediction']))
        print(mean_squared_error(datasource.df[target], predictions['prediction']))

        import matplotlib.pyplot as plt
        df = pd.read_csv('tests/data/sunspots.csv')
        true = df[target].values
        preds = predictions['prediction'].values
        plt.plot(true)
        plt.plot(preds)
        plt.show()

    def test_grouped_timeseries(self):
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/arrivals.csv')
        target = 'Traffic'
        predictor = predictor_from_problem(datasource.df, ProblemDefinition.from_dict({'target': target,
            'time_aim': 100,
            'nfolds': 4,
            'anomaly_detection': True,
            'timeseries_settings': {
                'order_by': ['T'],
                'group_by': ['Country'],
                'use_previous_target': True,
                'window': 5
            },
            }))

        predictor.learn(datasource.df)
        predictions = predictor.predict(datasource.df)

        # @TODO: Remove later
        print('R2 score:', r2_score(datasource.df[target], predictions['prediction']))
        print('MAE:', mean_absolute_error(datasource.df[target], predictions['prediction']))
        print('MSE:', mean_squared_error(datasource.df[target], predictions['prediction']))

        import matplotlib.pyplot as plt
        df = pd.read_csv('tests/data/arrivals.csv')
        true = df[target].values
        preds = predictions['prediction'].values
        plt.plot(true)
        plt.plot(preds)
        plt.show()

    def test_time_series_classification(self):
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/occupancy.csv')
        target = 'Occupancy'

        predictor = predictor_from_problem(
        datasource.df, ProblemDefinition.from_dict({'target': target,
                'time_aim': 100,
                'nfolds': 10,
                'anomaly_detection': False,
                'timeseries_settings': {
                    'order_by': ['date'],
                    'use_previous_target': True,
                    'window': 10
                },
                }))

        predictor.learn(datasource.df)
        predictions = predictor.predict(datasource.df)

        # @TODO: Remove later
        print('R2 score:', balanced_accuracy_score(datasource.df[target], predictions['prediction']))

        import matplotlib.pyplot as plt
        df = pd.read_csv('tests/data/occupancy.csv')
        true = df[target].values
        preds = predictions['prediction'].values
        plt.plot(true)
        plt.plot(preds)
        plt.show()

    def test_long_forecasts(self):
        pass

    def test_infer_mode(self):
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        group = 'Country'
        data = pd.read_csv('tests/data/arrivals.csv')
        train = pd.DataFrame(columns=data.columns)
        test = pd.DataFrame(columns=data.columns)

        train_ratio = 0.8
        for g in data[group].unique():
            subframe = data[data[group]==g]
            length = subframe.shape[0]
            train = train.append(subframe[:int(length*train_ratio)])
            test = test.append(subframe[int(length*train_ratio):])

        target = 'Traffic'
        predictor = predictor_from_problem(train, ProblemDefinition.from_dict({'target': target,
                                                                               'time_aim': 100,
                                                                               'nfolds': 4,
                                                                               'anomaly_detection': True,
                                                                               'timeseries_settings': {
                                                                                   'order_by': ['T'],
                                                                                   'group_by': ['Country'],
                                                                                   'nr_predictions': 3,
                                                                                   'use_previous_target': True,
                                                                                   'window': 5
                                                                               },
                                                                               }))
        predictor.learn(train)


        # features[-1][0] = 'make_predictions'  # add make_predictions column as mindsdb would
        # labels = [generate_timeseries_labels(features[:-1])]
        #
        # feature_headers = list(map(lambda col: col[0], features))
        # label_headers = list(map(lambda col: col[0], labels))
        #
        # # Create the training dataset and save it to a file
        # columns_train = list(map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        # columns_train.extend(list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        # columns_to_file(
        #     columns_train,
        #     train_file_name,
        #     headers=[*feature_headers, *label_headers]
        # )
        #
        # # force make_predictions column to be false, thus triggering inference for stream use cases
        # features[-1] = generate_value_cols(['false'], data_len, ts_hours * 3600)[0]
        # features[-1][0] = 'make_predictions'
        #
        # # Create the testing dataset and save it to a file
        # columns_test = list(map(lambda col: col[int(len(col) * 3 / 4):], features))
        # columns_to_file(
        #     columns_test,
        #     test_file_name,
        #     headers=feature_headers
        # )
        #
        # mdb = Predictor(name='test_timeseries_infer')
        #
        # mdb.learn(
        #     from_data=train_file_name,
        #     to_predict=label_headers,
        #     timeseries_settings={
        #         'order_by': [feature_headers[0]],
        #         'historical_columns': [feature_headers[-2]],
        #         'window': 3
        #     },
        #     stop_training_in_x_seconds=10,
        #     use_gpu=False,
        #     advanced_args={'debug': True}
        # )
        #
        # results = mdb.predict(when_data=test_file_name, use_gpu=False)
        #
        # # Check there is an additional row, which we inferred and then predicted for
        # assert len(results._data[label_headers[0]]) == len(columns_test[-2]) + 1
        # for row in results:
        #     expect_columns = [label_headers[0], label_headers[0] + '_confidence']
        #     for col in expect_columns:
        #         assert col in row
        #
        # for row in [x.explanation[label_headers[0]] for x in results]:
        #     assert row['confidence_interval'][0] <= row['predicted_value'] <= row['confidence_interval'][1]
        #
        # model_data = F.get_model_data('test_timeseries_infer')
        # assert model_data

