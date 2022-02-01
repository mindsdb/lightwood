from copy import deepcopy
from typing import List

import pandas as pd


class BaseImputer:
    def __init__(self, target: str, dependencies: List[str] = []):
        """
        Lightwood imputers will fill in missing values in any given column.
        
        The single method to implement, `impute`, is where the logic for filling in missing values has to be specified.
        
        Note that if access to other columns is required, this can be specified with the `dependencies` parameter.

        :param target: Column that the imputer will modify.
        :param dependencies: Any additional columns (other than the target) that will be needed inside `impute()`.
        """  # noqa
        self.target = target
        self.dependencies = dependencies

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Please implement your custom imputing logic in a module that inherits this class.")


class NumericalImputer(BaseImputer):
    def __init__(self, target: str, dependencies: List[str] = [], value: str = 'zero', typecast: str = None):
        """
        Imputer for numerical columns. Supports a handful of different approaches to define the imputation value.
        
        String to invoke this class from the cleaner is "numerical.$value", with "value" one of the valid options defined below.
        
        :param value: The value to impute. One of 'mean', 'median', 'zero', 'mode'.
        :param typecast: Used to cast the column into either 'int' or 'float' (`None` skips forced casting).
        """  # noqa
        self.value = value
        self.typecast = typecast
        super().__init__(target, dependencies)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = deepcopy(data)
        col = self.target

        if data[col].dtype not in (int, float):
            if self.typecast:
                try:
                    data[col] = data[col].astype(float)
                except ValueError:
                    raise Exception(f'Numerical imputer failed to cast column {col} to float!')
            else:
                raise Exception(f'Numerical imputer used in non-numeric column {col} with dtype {data[col].dtype}!')

        if self.value == 'mean':
            value = data[col].dropna().mean()
        elif self.value == 'median':
            value = data[col].dropna().median()
        elif self.value == 'mode':
            value = data[col].dropna().mode().iloc[0]  # if there's a tie, this chooses the smallest value
        else:
            value = 0.0

        data[col] = data[col].fillna(value=value)

        return data


class CategoricalImputer(BaseImputer):
    def __init__(self, target: str, dependencies: List[str] = [], value: str = 'zero'):
        """
        Imputer for categorical columns.
        
        String to invoke this class from the cleaner is "categorical.$value", with "value" one of the valid options defined below.
        
        :param value: One of 'mode', 'unk'. The former replaces missing data with the most common label, and the latter with an "UNK" string.
        """  # noqa
        self.value = value
        super().__init__(target, dependencies)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        data = deepcopy(data)
        col = self.target

        if self.value == 'mode':
            value = data[col].dropna().mode().iloc[0]
        else:
            value = 'UNK'

        data[col] = data[col].fillna(value=value)
        return data
