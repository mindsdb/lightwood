from typing import Dict
from lightwood.api import TypeInformation, StatisticalAnalysis, ProblemDefinition, dtype
import pandas as pd
import numpy as np
from lightwood.data.cleaner import _clean_float_or_none
from lightwood.helpers.numeric import filter_nan
from lightwood.helpers.seed import seed
from lightwood.data.cleaner import cleaner
from lightwood.helpers.log import log
from scipy.stats import entropy


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


def compute_entropy_biased_buckets(histogram):
    S, biased_buckets = None, None
    if histogram is not None:
        hist_x = histogram['x']
        hist_y = histogram['y']
        nr_values = sum(hist_y)
        S = entropy([x / nr_values for x in hist_y], base=max(2, len(hist_y)))
        if S < 0.25:
            pick_nr = -max(1, int(len(hist_y) / 10))
            biased_buckets = [hist_x[i] for i in np.array(hist_y).argsort()[pick_nr:]]
    return S, biased_buckets


def statistical_analysis(data: pd.DataFrame,
                         dtypes: Dict[str, str],
                         identifiers: Dict[str, object],
                         problem_definition: ProblemDefinition) -> StatisticalAnalysis:
    seed()
    log.info('Starting statistical analysis')
    df = cleaner(data, dtypes, problem_definition.pct_invalid, problem_definition.ignore_features, identifiers, problem_definition.target, 'train', problem_definition.timeseries_settings)
    
    missing = {col: len([x for x in df[col] if x is None]) / len(df[col]) for col in df.columns}
    distinct = {col: len(set(df[col])) / len(df[col]) for col in df.columns}

    nr_rows = len(df)
    target = problem_definition.target
    # get train std, used in analysis
    if dtypes[target] in [dtype.float, dtype.integer]:
        train_std = df[target].astype(float).std()
    elif dtypes[target] in [dtype.array]:
        try:
            all_vals = []
            for x in df[target]:
                all_vals += x
            train_std = pd.Series(all_vals).astype(float).std()
        except Exception as e:
            log.warning(e)
            train_std = 1.0
    else:
        train_std = 1.0

    histograms = {}
    # Get histograms for each column
    for col in df.columns:
        histograms[col] = None
        if dtypes[col] in (dtype.categorical, dtype.binary):
            hist = dict(df[col].value_counts().apply(lambda x: x / len(df[col])))
            histograms[col] = {
                'x': list(hist.keys()),
                'y': list(hist.values())
            }
        if dtypes[col] in (dtype.integer, dtype.float):
            histograms[col] = get_numeric_histogram(filter_nan(df[col]), dtypes[col])

    # get observed classes, used in analysis
    target_class_distribution = None
    if dtypes[target] in (dtype.categorical, dtype.binary):
        target_class_distribution = dict(df[target].value_counts().apply(lambda x: x / len(df[target])))
        train_observed_classes = list(target_class_distribution.keys())
    elif dtypes[target] == dtype.tags:
        train_observed_classes = None  # @TODO: pending call to tags logic -> get all possible tags
    else:
        train_observed_classes = None

    bias = {}
    for col in df.columns:
        S, biased_buckets = compute_entropy_biased_buckets(histograms[col])
        bias[col] = {
            'entropy': S,
            'description': """Under the assumption of uniformly distributed data (i.e., same probability for Head or Tails on a coin flip) mindsdb tries to detect potential divergences from such case, and it calls this "potential bias". Thus by our data having any potential bias mindsdb means any divergence from all categories having the same probability of being selected.""",
            'biased_buckets': biased_buckets
        }

    log.info('Finished statistical analysis')
    return StatisticalAnalysis(
        nr_rows=nr_rows,
        train_std_dev=train_std,
        train_observed_classes=train_observed_classes,
        target_class_distribution=target_class_distribution,
        histograms=histograms,
        missing=missing,
        distinct=distinct,
        bias=bias
    )
