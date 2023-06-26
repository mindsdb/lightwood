import unittest
import pandas as pd
from tests.utils.timing import train_and_check_time_aim
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import predictor_from_problem


class TestEmbeddingPredictor(unittest.TestCase):
    def test_0_embedding_at_inference_time(self):
        df = pd.read_csv('tests/data/hdi.csv')
        pdef = ProblemDefinition.from_dict({'target': 'Development Index', 'time_aim': 10})
        predictor = predictor_from_problem(df, pdef)
        train_and_check_time_aim(predictor, df, ignore_time_aim=True)
        predictions = predictor.predict(df, args={'return_embedding': True})

        self.assertTrue(predictions.shape[0] == len(df))
        self.assertTrue(predictions.shape[1] != 1)  # embedding dimension

    def test_1_embedding_only_at_creation(self):
        df = pd.read_csv('tests/data/hdi.csv')
        target = 'Development Index'
        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 10, 'embedding_only': True})
        predictor = predictor_from_problem(df, pdef)
        train_and_check_time_aim(predictor, df, ignore_time_aim=True)
        predictions = predictor.predict(df)

        self.assertTrue(predictions.shape[0] == len(df))
        self.assertTrue(predictions.shape[1] == predictor.ensemble.embedding_size)
        self.assertTrue(len(predictor.mixers) == 0)
