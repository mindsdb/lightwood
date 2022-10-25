# import unittest
# import pandas as pd

from lightwood.analysis import Robbie
# from lightwood.api.high_level import ProblemDefinition, json_ai_from_problem
# from lightwood.api.high_level import code_from_json_ai, predictor_from_code


class TestRobbie(unittest.TestCase):
    def test_0_robbie_analysis(self):
        if Robbie is None:
            print('Skipping Robbie test when values is empty')
            return
