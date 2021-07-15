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

    def test_long_forecasts_and_infer_mode(self):
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
                                                                                   'nr_predictions': 2,
                                                                                   'use_previous_target': True,
                                                                                   'window': 5
                                                                               },
                                                                               }))
        predictor.learn(train)

        # tests long forecasts
        preds = predictor.predict(test)

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = predictor.predict(test)
        print(preds)

        # # Check there is an additional row, which we inferred and then predicted for
        # assert len(results._data[label_headers[0]]) == len(columns_test[-2]) + 1
        # for row in results:
        #     expect_columns = [label_headers[0], label_headers[0] + '_confidence']
        #     for col in expect_columns:
        #         assert col in row
