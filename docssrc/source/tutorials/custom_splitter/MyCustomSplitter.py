
from lightwood.api.dtype import dtype
import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import product
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.log import log


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def MySplitter(
    data: pd.DataFrame,
    target: str,
    pct_train: float = 0.8,
    pct_dev: float = 0.1,
    seed: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Custom splitting function


    :param data: Input data
    :param target: Name of the target
    :param pct_train: Percentage of data reserved for training, taken out of full data
    :param pct_dev: Percentage of data reserved for dev, taken out of train data
    :param seed: Random seed for reproducibility

    :returns: A dictionary containing the keys train, test and dev with their respective data frames.
    """

    # Shuffle the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split into feature columns + target
    X = data.iloc[:, data.columns != target]  # .values
    y = data[target]  # .values

    # Create a train/test split
    X2, X_test, y2, y_test = train_test_split(
        X, y, train_size=pct_train, random_state=seed, stratify=data[target]
    )

    X_train, X_dev, y_train, y_dev = train_test_split(
        X2, y2, test_size=pct_dev, random_state=seed, stratify=y2
    )

    # Create a SMOTE model and bump up underbalanced class JUST for train data
    SMOTE_model = SMOTE(random_state=seed)

    Xtrain_mod, ytrain_mod = SMOTE_model.fit_resample(X_train, y_train.ravel())

    Xtrain_mod[target] = ytrain_mod
    X_test[target] = y_test
    X_dev[target] = y_dev

    return {"train": Xtrain_mod, "test": X_test, "dev": X_dev}
