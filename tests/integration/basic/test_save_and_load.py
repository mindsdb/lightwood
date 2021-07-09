from lightwood.api.high_level import code_from_problem, predictor_from_code
from lightwood.api.types import ProblemDefinition
import unittest
from mindsdb_datasources import FileDS


def save(predictor, path):
    predictor.save()


def train(predictor, df):
    predictor.learn(df)


def load(code, path):
    PredictorClass = predictor_from_code(code, return_class=True)
    predictor = PredictorClass.load(path)
    return predictor


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        # call: Go with dataframes
        df = FileDS('tests/data/adult.csv').df
        
        code = code_from_problem(df, ProblemDefinition.from_dict({'target': 'income', 'time_aim': 50}))
        predictor = predictor_from_code(code)

        save(predictor, 'a_path.pickle')
        predictor_1 = load(code, 'a_path.pickle')
        predictor_1.learn(df)

        save(predictor_1, 'a_path.pickle')
        predictor_2 = load(code, 'a_path.pickle')
        predictor_2.learn(df)
        print('Making predictions')
        predictions = predictor.predict(df.iloc[0:3])
        print(predictions)
