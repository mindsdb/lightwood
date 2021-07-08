from lightwood.api.types import ProblemDefinition
import unittest
import importlib


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        # call: Go with dataframes
        df = FileDS('tests/data/adult.csv').df
        
        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': 'income', 'time_aim': 50}))
        predictor.learn(df)

        print('Making predictions')
        predictions = predictor.predict(df.iloc[0:3])
        print(predictions)
