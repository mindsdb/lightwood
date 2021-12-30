from lightwood.api.high_level import code_from_problem, predictor_from_code, predictor_from_state
from lightwood.api.types import ProblemDefinition
import unittest
import multiprocessing as mp
import os
import pandas as pd


def save(predictor, path):
    predictor.save(path)


def train(predictor, df):
    predictor.learn(df)


def execute_first_bit(code, df, path):
    predictor = predictor_from_code(code)
    save(predictor, path)


def execute_second_bit(code, df, path):
    predictor_1 = predictor_from_state(path, code)
    predictor_1.learn(df)

    save(predictor_1, path)
    execute_third_bit(code, df, path)


def execute_third_bit(code, df, path):
    print('Running execute_third_bit')
    predictor_2 = predictor_from_state(path, code)
    predictions = predictor_2.predict(df.iloc[0:3])
    for p in predictions['prediction']:
        assert p is not None
    print('Done running third bit')


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        ctx = mp.get_context('spawn')
        df = pd.read_csv('tests/data/hdi.csv').iloc[0:400]
        code = code_from_problem(df, ProblemDefinition.from_dict({'target': 'Development Index', 'time_aim': 20}))
        path = 'test.pickle'
        try:
            os.remove(path)
        except Exception:
            pass
        proc = ctx.Process(target=execute_first_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()

        proc = ctx.Process(target=execute_second_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()

        proc = ctx.Process(target=execute_third_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()
