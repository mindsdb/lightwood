from lightwood.api.types import ProblemDefinition
import pandas as pd
import unittest


class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/boston.csv')
        
        import json
        from lightwood.api.high_level import json_ai_from_problem
        json.dump(json_ai_from_problem(df, ProblemDefinition.from_dict({'target': 'MEDV', 'time_aim': 200})).to_dict(), open('AI2.json', 'w'))

        predictor = predictor_from_problem(df, ProblemDefinition.from_dict({'target': 'MEDV', 'time_aim': 200}))
        predictor.learn(df)

        predictions = predictor.predict(df)
        print(predictions[0:100])
