import unittest
import importlib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, balanced_accuracy_score

from lightwood.api.types import ProblemDefinition
from lightwood.api import make_predictor


class TestTimeseries(unittest.TestCase):
    def test_timeseries_regression(self):
        """
        Tests a regression dataset and unsupervised anomaly detection
        """
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/sunspots.csv')
        target = 'Sunspots'
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': target,
                                                                              'time_aim': 100,
                                                                              'anomaly_detection': False,
                                                                              'timeseries_settings': {
                                                                                  'order_by': ['Month'],
                                                                                  'use_previous_target': True,
                                                                                  'window': 5
                                                                                },
                                                                              }),
                                                 datasource.df)

        with open('dynamic_predictor.py', 'w') as fp:
            fp.write(predictor_class_str)

        predictor_class = importlib.import_module('dynamic_predictor').Predictor
        print('Class was evaluated successfully')

        predictor = predictor_class()
        print('Class initialized successfully')

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
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/arrivals.csv')
        target = 'Traffic'
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': target,
                                                                              'time_aim': 100,
                                                                              'nfolds': 4,
                                                                              'anomaly_detection': True,
                                                                              'timeseries_settings': {
                                                                                  'order_by': ['T'],
                                                                                  'group_by': ['Country'],
                                                                                  'use_previous_target': True,
                                                                                  'window': 5
                                                                              },
                                                                              }),
                                                 datasource.df)

        with open('dynamic_predictor.py', 'w') as fp:
            fp.write(predictor_class_str)

        predictor_class = importlib.import_module('dynamic_predictor').Predictor
        print('Class was evaluated successfully')

        predictor = predictor_class()
        print('Class initialized successfully')

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
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/occupancy.csv')
        target = 'Occupancy'
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': target,
                                                                              'time_aim': 100,
                                                                              'nfolds': 10,
                                                                              'anomaly_detection': False,
                                                                              'timeseries_settings': {
                                                                                  'order_by': ['date'],
                                                                                  'use_previous_target': True,
                                                                                  'window': 10
                                                                              },
                                                                              }),
                                                 datasource.df)

        with open('dynamic_predictor.py', 'w') as fp:
            fp.write(predictor_class_str)

        predictor_class = importlib.import_module('dynamic_predictor').Predictor
        print('Class was evaluated successfully')

        predictor = predictor_class()
        print('Class initialized successfully')

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

    def test_stream_predictions(self):
        pass
