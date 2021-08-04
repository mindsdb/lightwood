from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import predictor_from_problem
from mindsdb_datasources import FileDS
import unittest


class TestText(unittest.TestCase):
    def test_0_train_and_predict_bypass(self):
        datasource = FileDS('tests/data/tripadvisor_binary_sample.csv')
        predictor = predictor_from_problem(datasource.df, ProblemDefinition.from_dict({'target': 'Label'}))
        predictor.learn(datasource.df)
        predictions = predictor.predict(datasource.df)
        for x in predictions['prediction']:
            assert x is not None

    def test_1_train_and_predict_model(self):
        datasource = FileDS('tests/data/wine_reviews_binary_sample.csv')
        predictor = predictor_from_problem(datasource.df, ProblemDefinition.from_dict({'target': 'label'}))
        predictor.learn(datasource.df)
        predictions = predictor.predict(datasource.df)
        for x in predictions['prediction']:
            assert x is not None
