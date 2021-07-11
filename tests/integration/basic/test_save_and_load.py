from lightwood.api.high_level import code_from_problem, predictor_from_code, predictor_from_state
from lightwood.api.types import ProblemDefinition
import unittest
from mindsdb_datasources import FileDS
import multiprocessing as mp
import os


def save(predictor, path):
    predictor.save(path)


def train(predictor, df):
    predictor.learn(df)


def execute_first_bit(code, df, path):
    predictor = predictor_from_code(code)
    save(predictor, path)


def execute_second_bit(code, df, path):
    predictor_1 = predictor_from_state(path, code)
    predictor_1.learn(data=df)

    save(predictor_1, path)
    execute_third_bit()


def execute_third_bit(code, df, path):
    predictor_2 = predictor_from_state(path, code)
    predictions = predictor_2.predict(df.iloc[0:3])
    for p in predictions['prediction']:
        assert p is not None


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        df = FileDS('tests/data/adult.csv').df.iloc[0:2000]
        code = code_from_problem(df, ProblemDefinition.from_dict({'target': 'income', 'time_aim': 30}))
        path = 'test.pickle'
        try:
            os.remove(path)
        except Exception:
            pass
        proc = mp.Process(target=execute_first_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()

        proc = mp.Process(target=execute_second_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()

        proc = mp.Process(target=execute_third_bit, args=(code, df, path,))
        proc.start()
        proc.join()
        proc.close()