from lightwood.api.high_level import code_from_problem, predictor_from_code, predictor_from_state
from lightwood.api.types import ProblemDefinition
import unittest
from mindsdb_datasources import FileDS
import sys
import multiprocessing as mp


def save(predictor, path):
    predictor.save(path)


def train(predictor, df):
    predictor.learn(df)


def execute_first_bit(code, df, path):
    predictor = predictor_from_code(code)

    save(predictor, path)


def execute_second_bit(code, df, path):    
    import gc
    try:
        del sys.modules['temp_predictor_module']
    except Exception:
        pass
    gc.collect()
    assert 'temp_predictor_module' not in sys.modules
    
    predictor_1 = predictor_from_state(path, code)
    print(predictor_1, predictor_1.learn)
    predictor_1.learn(data=df)

    save(predictor_1, path)


def execute_third_bit(code, df, path):
    predictor_2 = predictor_from_state(path, code)
    print('Making predictions')
    predictions = predictor_2.predict(df.iloc[0:3])
    print(predictions)


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        df = FileDS('tests/data/adult.csv').df
        code = code_from_problem(df, ProblemDefinition.from_dict({'target': 'income', 'time_aim': 300}))
        path = 'a_path.pickle'
        
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