import importlib

from tests.integration.helpers import ClickhouseTest, break_dataset
from lightwood.api.types import ProblemDefinition
from lightwood.api import make_predictor


class TestTimeseries(ClickhouseTest):
    def test_timeseries(self):
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        datasource = FileDS('tests/data/sunspots.csv')
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': 'Sunspots',
                                                                              'anomaly_detection': False,
                                                                              'use_previous_target': True,
                                                                              'timeseries_settings': {
                                                                                  'order_by': ['Month'],
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
        print(predictions[0:100])

    def test_grouped_timeseries(self):
        pass

    def test_anomaly_detection(self):
        pass