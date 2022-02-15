import unittest

import numpy as np
import pandas as pd

from lightwood.helpers.imputers import NumericalImputer, CategoricalImputer


class TestImputers(unittest.TestCase):
    def _load_df(self, cols_to_cat=[]):
        def _to_cat(x):
            return chr(int(x) + 97) if x == x else x

        df = pd.read_csv('tests/data/hdi.csv')
        for col in cols_to_cat:
            df[col] = df[col].apply(_to_cat).astype(str)

        np.random.seed(0)
        idxs = np.random.randint(0, high=len(df), size=(int(len(df) * 0.2)))  # drop some rows at random
        df.loc[idxs] = np.nan
        df.iloc[0] = np.nan  # force first row to be nan for filled value checks
        return df

    def test_numerical(self):
        target = 'Literacy (%)'
        df = self._load_df()
        for value, expected in zip(('mean', 'median', 'mode', 'zero'),
                                   (df[target].mean(), df[target].median(), df[target].dropna().mode().iloc[0], 0)):
            imp = NumericalImputer(target=target, value=value)
            ndf = imp.impute(df)

            assert df[target].isna().any()
            assert not ndf[target].isna().any()
            assert ndf[target].iloc[0] == expected

    def test_categorical(self):

        target = 'Development Index'
        df = self._load_df(cols_to_cat=[target])
        for value, expected in zip(('mode', ), (df[target].dropna().mode().iloc[0], )):
            imp = CategoricalImputer(target=target, value=value)
            ndf = imp.impute(df)

            assert df[target].isna().any()
            assert not ndf[target].isna().any()
            assert ndf[target].iloc[0] == expected
