from lightwood.api.types import ProblemDefinition
import unittest
import os
import importlib


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        # call: Go with dataframes
        datasource = FileDS('../data/boston.csv')
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': 'MEDV'}), datasource)

        try:
            with open('dynamic_predictor.py', 'w') as fp:
                fp.write(predictor_class_str)

            predictor_class = importlib.import_module('dynamic_predictor').Predictor
            print('Class was evaluated successfully')

            predictor = predictor_class()
            print('Class initialized successfully')

            predictor.learn(datasource)

            predictions = predictor.predict(datasource)
            print(predictions[0:100])
        finally:
            print('Not removing predictor for debugging purposes')
            #os.remove('dynamic_predictor.py')
