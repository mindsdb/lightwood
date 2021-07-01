from lightwood.api import TypeInformation, StatisticalAnalysis, ProblemDefinition, dtype
import pandas as pd
import numpy as np
from lightwood.data.cleaner import _clean_float_or_none
from lightwood.helpers.numeric import filter_nan
from lightwood.helpers.seed import seed


def get_numeric_histogram(data, data_dtype):
    data = [_clean_float_or_none(x) for x in data]
    Y, X = np.histogram(data, bins=min(50, len(set(data))),
                        range=(min(data), max(data)), density=False)
    if data_dtype == dtype.integer:
        Y, X = np.histogram(data, bins=[int(round(x)) for x in X], density=False)

    X = X[:-1].tolist()
    Y = Y.tolist()

    return {
        'x': X,
        'y': Y
    }


def statistical_analysis(data: pd.DataFrame,
                         type_information: TypeInformation,
                         problem_definition: ProblemDefinition) -> StatisticalAnalysis:
    seed()
    df = data
    
    nr_rows = len(df)
    target = problem_definition.target
    # get train std, used in analysis
    if type_information.dtypes[target] in [dtype.float, dtype.integer]:
        train_std = df[target].astype(float).std()
    else:
        train_std = None

    histograms = {}
    # Get histograms for each column
    for col in df.columns:
        if type_information.dtypes[col] in (dtype.categorical, dtype.binary):
            histograms[col] = dict(df[col].value_counts().apply(lambda x: x / len(df[col])))
        if type_information.dtypes[col] in (dtype.integer, dtype.float):
            histograms[col] = get_numeric_histogram(filter_nan(df[col]), type_information.dtypes[col])

    # get observed classes, used in analysis
    target_class_distribution = None
    if type_information.dtypes[target] in (dtype.categorical, dtype.binary):
        target_class_distribution = dict(df[target].value_counts().apply(lambda x: x / len(df[target])))
        train_observed_classes = list(target_class_distribution.keys())
    elif type_information.dtypes[target] == dtype.tags:
        train_observed_classes = None  # @TODO: pending call to tags logic -> get all possible tags
    else:
        train_observed_classes = None
    
    get_numeric_histogram

    return StatisticalAnalysis(
        nr_rows=nr_rows,
        train_std_dev=train_std,
        train_observed_classes=train_observed_classes,
        target_class_distribution=target_class_distribution,
        histograms=histograms
    )
