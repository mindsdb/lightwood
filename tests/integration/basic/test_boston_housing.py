from lightwood.api.types import ProblemDefinition
import unittest


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem
        from mindsdb_datasources import FileDS

        # call: Go with dataframes
        datasource = FileDS('tests/data/boston.csv')
        predictor = predictor_from_problem(datasource.df, ProblemDefinition.from_dict({'target': 'MEDV'}))
        predictor.learn(datasource.df)

        predictions = predictor.predict(datasource.df)
        print(predictions[0:100])
