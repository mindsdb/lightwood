import random
import unittest
import numpy as np
import pandas as pd
from typing import List
from lightwood.api.types import ProblemDefinition
from tests.utils.timing import train_and_check_time_aim
from lightwood.api.high_level import json_ai_from_problem, code_from_json_ai, predictor_from_code, predictor_from_problem  # noqa

np.random.seed(0)


class TestArrayTarget(unittest.TestCase):
    def test_0_num_array(self):
        """ Tests numerical array input and output. """
        # task: learn to emit next `arr_len` numbers given any numerical array input of length `arr_len`
        df = pd.DataFrame()
        arr_len = 4
        df['input'] = [[row + i for i in range(arr_len)] for row in range(200)]
        df['output'] = [[row + i + arr_len for i in range(arr_len)] for row in range(200)]

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': 'output',
                                                                        'time_aim': 80,
                                                                        }))
        predictor.learn(train)
        predictor.predict(test)


    def test_1_cat_array(self):
        """ Tests categorical array input and output. """
        # task: learn to reverse the `arr_len`-length input array.
        df = pd.DataFrame()
        arr_len = 4
        # chr(65 + int(str(x / 10000)[0]
        df['input'] = [[chr(65 + i + row % 4) for i in range(arr_len)] for row in range(200)]
        df['output'] = [[chr(65 + i + row % 4) for i in range(arr_len)][::-1] for row in range(200)]

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': 'output',
                                                                        'time_aim': 80,
                                                                        }))
        predictor.learn(train)
        predictor.predict(test)
