import unittest
import pandas as pd
from lightwood.api.high_level import code_from_json_ai, json_ai_from_problem, predictor_from_code

from lightwood.api.types import ProblemDefinition


class TestQuantumMixer(unittest.TestCase):
    # Interesting: has coordinates as inputs
    def test_0_predict_file_flow(self):

        df = pd.read_csv('tests/data/hdi.csv')

        target = 'Development Index'

        json_ai = json_ai_from_problem(df, ProblemDefinition.from_dict({
            'target': target,
            'time_aim': 80
        }))

        json_ai.outputs[target].mixers = [{
            'module': 'lightwood.mixer.QuantumMixer',
            'args': {
                'stop_after': '$problem_definition.seconds_per_mixer',
                'dtype_dict': '$dtype_dict',
                'target': '$target',
                'target_encoder': '$encoders[self.target]'
            }
        }]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)
        predictor.learn(df)
        predictions = predictor.predict(df)

        # sanity checks
        # self.assertTrue(accuracy_score(df[target], predictions['prediction'].astype(int)) > 0.5)
        print(df[target])
        print(predictions['prediction'])

        breakpoint()

        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))

