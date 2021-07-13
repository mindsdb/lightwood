"""
Evaluation of conformal predictors.
"""

# Original author: Henrik Linusson (github.com/donlnz)

# Note (Patricio): TODO: I am not sure we should include this one... it might be a good idea to run cross_val_score() depending on how the data source folds turn out

# TODO: cross_val_score/run_experiment should possibly allow multiple to be evaluated on identical folding


from lightwood.analysis.nc.base import RegressorMixin, ClassifierMixin

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.base import clone, BaseEstimator


class BaseIcpCvHelper(BaseEstimator):
    """Base class for cross validation helpers.
    """
    def __init__(self, icp, calibration_portion):
        super(BaseIcpCvHelper, self).__init__()
        self.icp = icp
        self.calibration_portion = calibration_portion

    def predict(self, x, significance=None):
        return self.icp.predict(x, significance)


class ClassIcpCvHelper(BaseIcpCvHelper, ClassifierMixin):
    """Helper class for running the ``cross_val_score`` evaluation
    method on IcpClassifiers.
    """
    def __init__(self, icp, calibration_portion=0.25):
        super(ClassIcpCvHelper, self).__init__(icp, calibration_portion)

    def fit(self, x, y):
        split = StratifiedShuffleSplit(y, n_iter=1,
                                       test_size=self.calibration_portion)
        for train, cal in split:
            self.icp.fit(x[train, :], y[train])
            self.icp.calibrate(x[cal, :], y[cal])


class RegIcpCvHelper(BaseIcpCvHelper, RegressorMixin):
    """Helper class for running the ``cross_val_score`` evaluation
    method on IcpRegressors.
    """
    def __init__(self, icp, calibration_portion=0.25):
        super(RegIcpCvHelper, self).__init__(icp, calibration_portion)

    def fit(self, x, y):
        split = train_test_split(x, y, test_size=self.calibration_portion)
        x_tr, x_cal, y_tr, y_cal = split[0], split[1], split[2], split[3]
        self.icp.fit(x_tr, y_tr)
        self.icp.calibrate(x_cal, y_cal)


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
def cross_val_score(model,x, y, iterations=10, folds=10, fit_params=None,
                    scoring_funcs=None, significance_levels=None,
                    verbose=False):
    """Evaluates a conformal predictor using cross-validation.

    Parameters
    ----------
    model : object
        Conformal predictor to evaluate.

    x : numpy array of shape [n_samples, n_features]
        Inputs of data to use for evaluation.

    y : numpy array of shape [n_samples]
        Outputs of data to use for evaluation.

    iterations : int
        Number of iterations to use for evaluation. The data set is randomly
        shuffled before each iteration.

    folds : int
        Number of folds to use for evaluation.

    fit_params : dictionary
        Parameters to supply to the conformal prediction object on training.

    scoring_funcs : iterable
        List of evaluation functions to apply to the conformal predictor in each
        fold. Each evaluation function should have a signature
        ``scorer(prediction, y, significance)``.

    significance_levels : iterable
        List of significance levels at which to evaluate the conformal
        predictor.

    verbose : boolean
        Indicates whether to output progress information during evaluation.

    Returns
    -------
    scores : pandas DataFrame
        Tabulated results for each iteration, fold and evaluation function.
    """

    fit_params = fit_params if fit_params else {}
    significance_levels = (significance_levels if significance_levels
                                                  is not None else np.arange(0.01, 1.0, 0.01))

    df = pd.DataFrame()

    columns = ['iter',
               'fold',
               'significance',
               ] + [f.__name__ for f in scoring_funcs]
    for i in range(iterations):
        idx = np.random.permutation(y.size)
        x, y = x[idx, :], y[idx]
        cv = KFold(y.size, folds)
        for j, (train, test) in enumerate(cv):
            if verbose:
                sys.stdout.write('\riter {}/{} fold {}/{}'.format(
                    i + 1,
                    iterations,
                    j + 1,
                    folds
                ))
            m = clone(model)
            m.fit(x[train, :], y[train], **fit_params)
            prediction = m.predict(x[test, :], significance=None)
            for k, s in enumerate(significance_levels):
                scores = [scoring_func(prediction, y[test], s)
                          for scoring_func in scoring_funcs]
                df_score = pd.DataFrame([[i, j, s] + scores],
                                        columns=columns)
                df = df.append(df_score, ignore_index=True)

    return df


def run_experiment(models, csv_files, iterations=10, folds=10, fit_params=None,
                   scoring_funcs=None, significance_levels=None,
                   normalize=False, verbose=False, header=0):
    """Performs a cross-validation evaluation of one or several conformal
    predictors on a	collection of data sets in csv format.

    Parameters
    ----------
    models : object or iterable
        Conformal predictor(s) to evaluate.

    csv_files : iterable
        List of file names (with absolute paths) containing csv-data, used to
        evaluate the conformal predictor.

    iterations : int
        Number of iterations to use for evaluation. The data set is randomly
        shuffled before each iteration.

    folds : int
        Number of folds to use for evaluation.

    fit_params : dictionary
        Parameters to supply to the conformal prediction object on training.

    scoring_funcs : iterable
        List of evaluation functions to apply to the conformal predictor in each
        fold. Each evaluation function should have a signature
        ``scorer(prediction, y, significance)``.

    significance_levels : iterable
        List of significance levels at which to evaluate the conformal
        predictor.

    verbose : boolean
        Indicates whether to output progress information during evaluation.

    Returns
    -------
    scores : pandas DataFrame
        Tabulated results for each data set, iteration, fold and
        evaluation function.
    """
    df = pd.DataFrame()
    if not hasattr(models, '__iter__'):
        models = [models]

    for model in models:
        is_regression = model.get_problem_type() == 'regression'

        n_data_sets = len(csv_files)
        for i, csv_file in enumerate(csv_files):
            if verbose:
                print('\n{} ({} / {})'.format(csv_file, i + 1, n_data_sets))
            data = pd.read_csv(csv_file, header=header)
            x, y = data.values[:, :-1], data.values[:, -1]
            x = np.array(x, dtype=np.float64)
            if normalize:
                if is_regression:
                    y = y - y.min() / (y.max() - y.min())
                else:
                    for j, y_ in enumerate(np.unique(y)):
                        y[y == y_] = j

            scores = cross_val_score(model, x, y, iterations, folds,
                                     fit_params, scoring_funcs,
                                     significance_levels, verbose)

            ds_df = pd.DataFrame(scores)
            ds_df['model'] = model.__class__.__name__
            try:
                ds_df['data_set'] = csv_file.split('/')[-1]
            except:
                ds_df['data_set'] = csv_file

            df = df.append(ds_df)

    return df

