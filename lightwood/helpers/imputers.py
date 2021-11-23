from copy import deepcopy

import pandas as pd


class BaseImputer:
    def __init__(self, target_col: str, value: str = 'zero', force_typecast: str = None):
        """
        Lightwood imputers can modify a subset of columns (typically, a single column) after the raw data has been cleaned.
        
        The single method to implement, `impute` is where the logic for missing values in all relevant columns has to be specified. The recommendation is to deepcopy the data frame prior to imputing.
        
        :param target_col: Column that the imputer will handle. 
        :param value: Specifies the imputation value.
        :param force_typecast: Setting this flag to something other than 'None' will force the column to be casted into either 'int' or 'float' 
        """  # noqa
        self.target_col = target_col
        self.value = value
        self.force_typecast = force_typecast

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Please implement your custom imputing logic in a module that inherits this class.")


class NumericalImputer(BaseImputer):
    def __init__(self, target_col: str, value: str = 'zero', force_typecast: str = None):
        """
        Imputer for numerical columns. Supports a handful of different approaches to define the imputation value.
        
        :param value: One of 'mean', 'median', 'zero', 'mode'.
        """  # noqa
        super().__init__(target_col, value, force_typecast)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = deepcopy(data)
        col = self.target_col

        if data[col].dtype not in (int, float):
            if self.force_typecast:
                try:
                    data[col] = data[col].astype(float)
                except ValueError:
                    raise Exception(f'Numerical imputer failed to cast column {col} to float!')
            else:
                raise Exception(f'Numerical imputer used in non-numeric column {col} with dtype {data[col].dtype}!')

        if self.value == 'mean':
            value = data[col].mean()
        elif self.value == 'median':
            value = data[col].median()
        elif self.value == 'mode':
            value = data[col].dropna().mode().iloc[0]  # if there's a tie, this chooses the smallest value
        else:
            value = 0

        data[col] = data[col].fillna(value=value)

        return data


class CategoricalImputer(BaseImputer):
    def __init__(self, target_col: str, value: str = 'zero'):
        """
        Imputer for categorical columns.
        
        :param value: One of 'mode', 'unk'. The former replaces missing data with the most common label, and the latter with an "UNK" string. 
        """  # noqa
        super().__init__(target_col, value)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = deepcopy(data)
        col = self.target_col

        if self.value == 'mode':
            value = data[col].dropna().mode().iloc[0]
        else:
            value = 'UNK'

        data[col] = data[col].fillna(value=value)
        return data
