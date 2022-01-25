import unittest
import numpy as np
import pandas as pd

from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai


class TestCleaner(unittest.TestCase):
    def test_0_imputers(self):

        df = pd.read_csv('tests/data/hdi.csv')[:100]
        df = df.rename(columns={'GDP ($ per capita)': 'GDP', 'Area (sq. mi.)': 'Area', 'Literacy (%)': 'Literacy'})
        target = 'Development Index'
        df['Infant mortality '] = df['Infant mortality '].apply(lambda x: 'High' if x >= 20 else 'Low')
        jai = json_ai_from_problem(df, ProblemDefinition.from_dict({'target': target, 'time_aim': 10}))

        # define columns to impute
        cat_mode_impute_col = 'Infant mortality '
        cat_unk_impute_col = 'Literacy'
        num_mean_impute_col = 'Population'
        num_mode_impute_col = 'GDP'
        num_zero_impute_col = 'Pop. Density '
        num_median_impute_col = 'Area'
        cols = [num_mean_impute_col, num_mode_impute_col, num_zero_impute_col, num_median_impute_col,
                cat_mode_impute_col, cat_unk_impute_col]

        # replace first row values with nans
        for col in cols:
            df[col].iloc[0] = np.nan

        num_mean_target_value = df[num_mean_impute_col].iloc[1:].mean()
        num_mode_target_value = df[num_mode_impute_col].iloc[1:].mode().iloc[0]
        num_median_target_value = df[num_median_impute_col].iloc[1:].median()
        num_zero_target_value = 0.0
        cat_mode_target_value = df[cat_mode_impute_col].iloc[1:].mode().iloc[0]
        cat_unk_target_value = 'UNK'

        jai.cleaner = {
            "module": "cleaner",
            "args": {
                "imputers": {
                    num_mean_impute_col: {"module": "NumericalImputer", "args": {"mode": "mean"}},
                    num_mode_impute_col: {"module": "NumericalImputer", "args": {"mode": "mode"}},
                    num_median_impute_col: {"module": "NumericalImputer", "args": {"mode": "median"}},
                    num_zero_impute_col: {"module": "NumericalImputer", "args": {"mode": "zero"}},
                    cat_mode_impute_col: {"module": "CategoricalImputer", "args": {"mode": "mode"}},
                    cat_unk_impute_col: {"module": "CategoricalImputer", "args": {"mode": "unk"}},
            }}
        }
        predictor = predictor_from_json_ai(jai)
        cleaned_data = predictor.preprocess(df)

        # check cleaner was assigned imputers
        assert jai.cleaner['args']['imputers'][num_mean_impute_col] == 'numerical.mean'
        assert jai.cleaner['args']['imputers'][num_mode_impute_col] == 'numerical.mode'
        assert jai.cleaner['args']['imputers'][num_median_impute_col] == 'numerical.median'
        assert jai.cleaner['args']['imputers'][num_zero_impute_col] == 'numerical.zero'
        assert jai.cleaner['args']['imputers'][cat_mode_impute_col] == 'categorical.mode'
        assert jai.cleaner['args']['imputers'][cat_unk_impute_col] == 'categorical.unk'

        # check data was imputed correctly
        assert cleaned_data[num_mean_impute_col].iloc[0] == num_mean_target_value
        assert cleaned_data[num_mode_impute_col].iloc[0] == num_mode_target_value
        assert cleaned_data[num_zero_impute_col].iloc[0] == num_zero_target_value
        assert cleaned_data[num_median_impute_col].iloc[0] == num_median_target_value
        assert cleaned_data[cat_mode_impute_col].iloc[0] == cat_mode_target_value
        assert cleaned_data[cat_unk_impute_col].iloc[0] == cat_unk_target_value
