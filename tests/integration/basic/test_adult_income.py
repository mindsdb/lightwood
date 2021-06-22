from lightwood.api.types import ProblemDefinition
import unittest
import importlib


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        # call: Go with dataframes
        df = FileDS('tests/data/adult.csv').df
        predictor_class_str = generate_predictor(ProblemDefinition.from_dict({'target': 'income', 'time_aim': 50}), df)

        with open('dynamic_predictor.py', 'w') as fp:
            fp.write(predictor_class_str)

        predictor_class = importlib.import_module('dynamic_predictor').Predictor
        print('Class was evaluated successfully')

        predictor = predictor_class()
        print('Class initialized successfully')

        predictor.learn(df)

        print('Making predictions')
        predictions = predictor.predict(df.iloc[0:3])
        print(predictions)
