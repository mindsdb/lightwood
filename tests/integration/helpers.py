import os
import json
import random
import unittest
import numpy as np


_var_name = 'DATABASE_CREDENTIALS_STRINGIFIED_JSON'
_var_value = os.getenv(_var_name)
if _var_value is None:
    with open(os.path.join(os.path.expanduser("~"), '.mindsdb_credentials.json'), 'r') as fp:
        _var_value = fp.read()

assert _var_value is not None, _var_name + ' ' + 'is not set'

DB_CREDENTIALS = json.loads(_var_value)


def break_dataset(df, null_cell_pct=0.2, stringify_numbers_pct=0.5):
    df = df.copy()

    n_rows, n_cols = df.shape

    # Make cells null
    for i in range(n_rows):
        for j in range(n_cols):
            if random.random() <= null_cell_pct:
                df.iloc[i, j] = np.nan

    # Stringify numbers
    for i in range(n_rows):
        for j in range(n_cols):
            if isinstance(df.iloc[i, j], (float, int)):
                if random.random() <= stringify_numbers_pct:
                    df.iloc[i, j] = str(df.iloc[i, j])

    return df


class ClickhouseTest(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['clickhouse']['user']
        self.PASSWORD = DB_CREDENTIALS['clickhouse']['password']
        self.HOST = DB_CREDENTIALS['clickhouse']['host']
        self.PORT = int(DB_CREDENTIALS['clickhouse']['port'])
        self.DATABASE = 'test_data'
