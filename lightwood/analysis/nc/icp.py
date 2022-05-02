"""
Inductive conformal predictors.
"""
# Original author: Henrik Linusson (github.com/donlnz)
from collections import defaultdict
from functools import partial
from typing import Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
from lightwood.analysis.nc.base import RegressorMixin, ClassifierMixin, TSMixin
from types import FunctionType


# -----------------------------------------------------------------------------
# Base inductive conformal predictor
# -----------------------------------------------------------------------------
class BaseIcp(BaseEstimator):
    """Base class for inductive conformal predictors.
    """

    def __init__(self, nc_function: FunctionType, condition: Union[bool, FunctionType] = None, cal_size: int = None):
        self.cal_x, self.cal_y = None, None
        self.nc_function = nc_function
        self.cal_size = cal_size  # if specified, defines size of calibration set

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        def default_condition(x):
            return 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

    def fit(self, x: np.array, y: np.array) -> None:
        """Fit underlying nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the nonconformity scorer.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the nonconformity scorer.

        Returns
        -------
        None
        """
        # TODO: incremental?
        self.nc_function.fit(x, y)

    def calibrate(self, x, y, increment=False):
        """Calibrate conformal predictor based on underlying nonconformity
        scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for calibrating the conformal predictor.

        y : numpy array of shape [n_samples, n_features]
            Outputs of examples for calibrating the conformal predictor.

        increment : boolean
            If ``True``, performs an incremental recalibration of the conformal
            predictor. The supplied ``x`` and ``y`` are added to the set of
            previously existing calibration examples, and the conformal
            predictor is then calibrated on both the old and new calibration
            examples.

        Returns
        -------
        None
        """
        self._calibrate_hook(x, y, increment)
        self._update_calibration_set(x, y, increment)

        if self.conditional:
            category_map = np.array([self.condition((x[i, :], y[i]))
                                     for i in range(y.size)])
            self.categories = np.unique(category_map)
            self.cal_scores = defaultdict(partial(np.ndarray, 0))

            for cond in self.categories:
                idx = category_map == cond
                cal_scores = self.nc_function.score(self.cal_x[idx, :],
                                                    self.cal_y[idx])
                self.cal_scores[cond] = np.sort(cal_scores)[::-1]
        else:
            self.categories = np.array([0])
            cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
            self.cal_scores = {0: np.sort(cal_scores)[::-1]}

        if self.cal_size:
            self.cal_scores = self._reduce_scores()

    def _reduce_scores(self):
        return {k: cs[::int(len(cs) / self.cal_size) + 1] for k, cs in self.cal_scores.items()}

    def _update_calibration_set(self, x: np.array, y: np.array, increment: bool) -> None:
        if increment and self.cal_x is not None and self.cal_y is not None:
            self.cal_x = np.vstack([self.cal_x, x])
            self.cal_y = np.hstack([self.cal_y, y])
        else:
            self.cal_x, self.cal_y = x, y

    def _calibrate_hook(self, x: np.array, y: np.array, increment: bool) -> None:
        pass


# -----------------------------------------------------------------------------
# Inductive conformal classifier
# -----------------------------------------------------------------------------
class IcpClassifier(BaseIcp, ClassifierMixin):
    """Inductive conformal classifier.

    Parameters
    ----------
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity of
        calibration examples and test patterns. Should implement ``fit(x, y)``
        and ``calc_nc(x, y)``.

    smoothing : boolean
        Decides whether to use stochastic smoothing of p-values.

    Attributes
    ----------
    cal_x : numpy array of shape [n_cal_examples, n_features]
        Inputs of calibration set.

    cal_y : numpy array of shape [n_cal_examples]
        Outputs of calibration set.

    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity scores.

    classes : numpy array of shape [n_classes]
        List of class labels, with indices corresponding to output columns
         of IcpClassifier.predict()

    See also
    --------
    IcpRegressor

    References
    ----------
    .. [1] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
        intervals with regression neural networks. Neural Networks, 24(8),
        842-851.
    """

    def __init__(self, nc_function: FunctionType, condition: Union[bool, FunctionType] = None, cal_size: int = None,
                 smoothing: bool = True) -> None:
        super(IcpClassifier, self).__init__(nc_function, condition, cal_size)
        self.classes = None
        self.smoothing = smoothing

    def _calibrate_hook(self, x: np.array, y: np.array, increment: bool = False) -> None:
        self._update_classes(y, increment)

    def _update_classes(self, y: np.array, increment: bool) -> None:
        if self.classes is None or not increment:
            self.classes = np.unique(y)
        else:
            self.classes = np.unique(np.hstack([self.classes, y]))

    def predict(self, x: np.array, significance: Optional[float] = None) -> np.array:
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        significance : float or None
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then the p-values
            are output rather than the predictions.

        Returns
        -------
        p : numpy array of shape [n_samples, n_classes]
            If significance is ``None``, then p contains the p-values for each
            sample-class pair; if significance is a float between 0 and 1, then
            p is a boolean array denoting which labels are included in the
            prediction sets.
        """
        # TODO: if x == self.last_x ...
        n_test_objects = x.shape[0]
        p = np.zeros((n_test_objects, self.classes.size))

        for i, c in enumerate(self.classes):
            test_class = np.zeros(x.shape[0], dtype=self.classes.dtype)
            test_class.fill(c)

            # TODO: maybe calculate p-values using cython or similar
            # TODO: interpolated p-values

            # TODO: nc_function.calc_nc should take X * {y1, y2, ... ,yn}
            test_nc_scores = self.nc_function.score(x, test_class)
            for j, nc in enumerate(test_nc_scores):
                cal_scores = self.cal_scores[self.condition((x[j, :], c))][::-1]
                n_cal = cal_scores.size
                n_eq = sum(np.where(cal_scores == nc, 1, 0))
                n_gt = sum(np.where(cal_scores > nc, 1, 0))

                if self.smoothing:
                    p[j, i] = (n_gt + n_eq * np.random.uniform(0, 1, 1)) / (n_cal + 1)
                else:
                    p[j, i] = (n_gt + n_eq) / (n_cal + 1)

        if significance is not None:
            return p > significance
        else:
            return p

    def predict_conf(self, x):
        """Predict the output values for a set of input patterns, using
        the confidence-and-credibility output scheme.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        Returns
        -------
        p : numpy array of shape [n_samples, 3]
            p contains three columns: the first column contains the most
            likely class for each test pattern; the second column contains
            the confidence in the predicted class label, and the third column
            contains the credibility of the prediction.
        """
        p = self.predict(x, significance=None)
        label = p.argmax(axis=1)
        credibility = p.max(axis=1)
        for i, idx in enumerate(label):
            p[i, idx] = -np.inf
        confidence = 1 - p.max(axis=1)

        return np.array([label, confidence, credibility]).T


# -----------------------------------------------------------------------------
# Inductive conformal regressor
# -----------------------------------------------------------------------------
class IcpRegressor(BaseIcp, RegressorMixin):
    """Inductive conformal regressor.

    Parameters
    ----------
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity of
        calibration examples and test patterns. Should implement ``fit(x, y)``,
        ``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

    Attributes
    ----------
    cal_x : numpy array of shape [n_cal_examples, n_features]
        Inputs of calibration set.

    cal_y : numpy array of shape [n_cal_examples]
        Outputs of calibration set.

    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity scores.

    See also
    --------
    IcpClassifier

    References
    ----------
    .. [1] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
        Inductive confidence machines for regression. In Machine Learning: ECML
        2002 (pp. 345-356). Springer Berlin Heidelberg.

    .. [2] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
        intervals with regression neural networks. Neural Networks, 24(8),
        842-851.
    """

    def __init__(self, nc_function: FunctionType, condition: bool = None, cal_size: int = None) -> None:
        super(IcpRegressor, self).__init__(nc_function, condition, cal_size)

    def predict(self, x: np.array, significance: bool = None) -> np.array:
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99}
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """
        # TODO: interpolated p-values

        n_significance = (99 if significance is None
                          else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], 2))

        condition_map = np.array([self.condition((x[i, :], None))
                                  for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             self.cal_scores[condition],
                                             significance)
                if n_significance > 1:
                    prediction[idx, :, :] = p
                else:
                    prediction[idx, :] = p

        return prediction


# -----------------------------------------------------------------------------
# Inductive conformal forecasting
# -----------------------------------------------------------------------------
class IcpTSRegressor(BaseIcp, TSMixin):
    """Inductive conformal forecasting.

    Parameters
    ----------
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity of
        calibration examples and test patterns. Should implement ``fit(x, y)``,
        ``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

    Attributes
    ----------
    cal_x : numpy array of shape [n_cal_examples, n_features]
        Inputs of calibration set.

    cal_y : numpy array of shape [n_cal_examples, horizon_length]
        Outputs of calibration set.

    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity scores.
    """

    def __init__(self, nc_function: FunctionType, horizon_length, condition: bool = None, cal_size: int = None) -> None:
        super(IcpTSRegressor, self).__init__(nc_function, condition, cal_size)
        self.horizon_length = horizon_length

    def calibrate(self, x, y, increment=False):
        """ 
        After calibration, handles incomplete target information by imputing the row-wise mean.
        """  # noqa
        super(IcpTSRegressor, self).calibrate(x, y, increment)
        for k, v in self.cal_scores.items():
            row_mean = np.nanmean(v, axis=1)
            idxs = np.where(np.isnan(v))
            v[idxs] = np.take(row_mean, idxs[0])
            self.cal_scores[k] = v

    def predict(self, x: np.array, significance: bool = None) -> np.array:
        """Predict the output values for a set of input patterns.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, horizon_length, 2] or [n_samples, horizon_length, 2, 99}
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each step of the test pattern, and each
            significance level (0.01, 0.02, ..., 0.99). If significance is a
            float between 0 and 1, then p contains the prediction intervals
            (minimum and maximum boundaries) for the all steps of the test
            patterns at the chosen significance level.
        """
        n_significance = (99 if significance is None else np.array(significance).size)

        if n_significance > 1:
            prediction = np.zeros((x.shape[0], self.horizon_length, 2, n_significance))
        else:
            prediction = np.zeros((x.shape[0], self.horizon_length, 2))

        condition_map = np.array([self.condition((x[i, :], None)) for i in range(x.shape[0])])

        for condition in self.categories:
            idx = condition_map == condition
            if np.sum(idx) > 0:
                p = self.nc_function.predict(x[idx, :],
                                             self.cal_scores[condition],
                                             significance)
                if n_significance > 1:
                    prediction[idx, :, :, :] = p
                else:
                    prediction[idx, :, :] = p

        return prediction
