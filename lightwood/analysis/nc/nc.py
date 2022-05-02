"""
Nonconformity functions.
"""

# Original author: Henrik Linusson (github.com/donlnz)

import abc
import numpy as np
import sklearn.base
from scipy.interpolate import interp1d
from copy import deepcopy


# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------
class ClassificationErrFunc(object):
    """Base class for classification model error functions.
    """ # noqa

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(ClassificationErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """ # noqa
        pass


class RegressionErrFunc(object):
    """Base class for regression model error functions.
    """ # noqa

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  # , norm=None, beta=0):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """ # noqa
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, 2]
            Minimum and maximum interval boundaries for each prediction.
        """ # noqa
        pass


class TSErrFunc(object):
    """Base class for time series model error functions.
    """ # noqa

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(TSErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, horizon_length]
            Forecasts for each sample.

        y : numpy array of shape [n_samples, horizon_length]
            True output series for each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples, horizon_length]
            Nonconformity scores of the samples.
        """ # noqa
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples, horizon_length]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, horizon_length, 2]
            Minimum and maximum interval boundaries for each step of the forecast.
        """ # noqa
        pass


class InverseProbabilityErrFunc(ClassificationErrFunc):
    """Calculates the probability of not predicting the correct class.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        1 - hat{P}(y_i | x) , .
    """ # noqa

    def __init__(self):
        super(InverseProbabilityErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
        return 1 - prob


class MarginErrFunc(ClassificationErrFunc):
    """
    Calculates the margin error.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        0.5 - frac{hat{P}(y_i | x) - max_{y , != , y_i} hat{P}(y | x)}{2}
    """ # noqa

    def __init__(self):
        super(MarginErrFunc, self).__init__()

    def apply(self, prediction, y):
        prediction = deepcopy(prediction).astype(float)
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
                prediction[i, int(y_)] = -np.inf
        return 0.5 - ((prob - prediction.max(axis=1)) / 2)


class AbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems.

        For each correct output in ``y``, nonconformity is defined as

        .. math::
            | y_i - hat{y}_i |
    """ # noqa

    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        # TODO: should probably warn against too few calibration examples
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class BoostedAbsErrorErrFunc(RegressionErrFunc):
    """ Calculates absolute error nonconformity for regression problems. Applies linear interpolation
    for nonconformity scores when we have less than 100 samples in the validation dataset.
    """ # noqa

    def __init__(self):
        super(BoostedAbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        if 1 < nc.size < 100:
            x = np.arange(nc.shape[0])
            interp = interp1d(x, nc)
            nc = interp(np.linspace(0, nc.size - 1, 100))
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
    """Calculates signed error nonconformity for regression problems.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        y_i - hat{y}_i

    References
    ----------
    .. [1] Linusson, Henrik, Ulf Johansson, and Tuve Lofstrom.
        Signed-error conformal regression. Pacific-Asia Conference on Knowledge
        Discovery and Data Mining. Springer International Publishing, 2014.
    """ # noqa

    def __init__(self):
        super(SignErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return (prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        upper = int(np.floor((significance / 2) * (nc.size + 1)))
        lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
        # TODO: should probably warn against too few calibration examples
        upper = min(max(upper, 0), nc.size - 1)
        lower = max(min(lower, nc.size - 1), 0)
        return np.vstack([-nc[lower], nc[upper]])


class TSAbsErrorErrFunc(TSErrFunc):
    """Calculates absolute error nonconformity for time series problems.

        For each forecasted step ``y_h`` for h \in 1..horizon, nonconformity is defined as

        .. math::
            | y_ni - hat{y}_ni |
            
        Following Stankeviciute, K. (2021). Conformal Time-Series Forecasting, we perform a 
        Bonferroni correction over the nonoconformity scores when applying the inverse function.
    """ # noqa

    def __init__(self, horizon_length):
        super(TSAbsErrorErrFunc, self).__init__()
        self.horizon_length = horizon_length

    def apply(self, prediction, y):
        """ calculate absolute error, eq. (6) in the paper """  # noqa
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        significance /= self.horizon_length  # perform Bonferroni correction, eq. (7) in the paper
        nc = np.sort(nc, axis=0)[::-1]
        border = int(np.floor(significance * (nc.shape[0] + 1))) - 1
        border = min(max(border, 0), nc.shape[0] - 1)
        return nc[border]


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class RegressorNormalizer(BaseScorer):
    def __init__(self, base_model, normalizer_model, err_func):
        super(RegressorNormalizer, self).__init__()
        self.base_model = base_model
        self.normalizer_model = normalizer_model
        self.err_func = err_func

    def fit(self, x, y):
        residual_prediction = self.base_model.predict(x)
        residual_error = np.abs(self.err_func.apply(residual_prediction, y))
        residual_error += 0.00001  # Add small term to avoid log(0)
        log_err = np.log(residual_error)
        self.normalizer_model.fit(x, log_err)

    def score(self, x, y=None):
        norm = np.exp(self.normalizer_model.predict(x))
        return norm


class BaseModelNc(BaseScorer):
    """Base class for nonconformity scorers based on an underlying model.

    Parameters
    ----------
    model : ClassifierAdapter or RegressorAdapter or TSAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc or RegressionErrFunc or TSErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.
    """ # noqa

    def __init__(self, model, err_func, normalizer=None, beta=0):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if (self.normalizer is not None and
                hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        """Fits the underlying model of the nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the underlying model.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the underlying model.

        Returns
        -------
        None
        """ # noqa
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """ # noqa
        prediction = self.model.predict(x)

        err = self.err_func.apply(prediction, y)
        if self.normalizer is not None:
            try:
                norm = self.normalizer.score(x) + self.beta
                err = err / norm
            except Exception:
                pass

        return err

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['model', 'normalizer']:  # model should not be copied
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result


# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ClassifierNc(BaseModelNc):
    """Nonconformity scorer using an underlying class probability estimating
    model.

    Parameters
    ----------
    model : ClassifierAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : ClassifierAdapter
        Underlying model object.

    err_func : ClassificationErrFunc
        Scorer function used to calculate nonconformity scores.
    """ # noqa

    def __init__(self,
                 model,
                 err_func=MarginErrFunc(),
                 normalizer=None,
                 beta=0):
        super(ClassifierNc, self).__init__(model,
                                           err_func,
                                           normalizer,
                                           beta)


# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseModelNc):
    """Nonconformity scorer using an underlying regression model.

    Parameters
    ----------
    model : RegressorAdapter
        Underlying regression model used for calculating nonconformity scores.

    err_func : RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : RegressorAdapter
        Underlying model object.

    err_func : RegressionErrFunc
        Scorer function used to calculate nonconformity scores.
    """ # noqa

    def __init__(self,
                 model,
                 err_func=AbsErrorErrFunc(),
                 normalizer=None,
                 beta=0):
        super(RegressorNc, self).__init__(model,
                                          err_func,
                                          normalizer,
                                          beta)

    def predict(self, x, nc, significance=None):
        """Constructs prediction intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        prediction, resulting in a prediction interval for each test pattern.

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
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """ # noqa
        n_test = x.shape[0]
        prediction = self.model.predict(x)

        norm = 1
        if self.normalizer is not None:
            try:
                norm = self.normalizer.score(x) + self.beta
            except Exception:
                pass

        if significance:
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            err_dist *= norm

            intervals = np.zeros((x.shape[0], 2))
            intervals[:, 0] = prediction - err_dist[0, :]
            intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


# -----------------------------------------------------------------------------
# Time series nonconformity scorers
# -----------------------------------------------------------------------------
class TSNc(BaseModelNc):
    """Nonconformity scorer using an underlying time series model.

    Parameters
    ----------
    model : TSAdapter
        Underlying regression model used for calculating nonconformity scores.

    err_func : TSErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : TSAdapter
        Underlying model object.

    err_func : TSErrFunc
        Scorer function used to calculate nonconformity scores.
    """ # noqa

    def __init__(self,
                 model,
                 err_func=TSAbsErrorErrFunc(horizon_length=1),
                 normalizer=None,
                 beta=0):
        super(TSNc, self).__init__(model,
                                   err_func,
                                   normalizer,
                                   beta)

    def predict(self, x, nc, significance=None):
        """Constructs forecast intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        step of the forecast, resulting in a prediction interval for each 
        step of the test pattern.

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
        p : numpy array of shape [n_samples, horizon_length, 2] or [n_samples, horizon_length, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each step of the test pattern, and each 
            significance level (0.01, 0.02, ..., 0.99). If significance is a 
            float between 0 and 1, then p contains the prediction intervals 
            (minimum and maximum boundaries) for the all steps of the test 
            patterns at the chosen significance level.
        """ # noqa
        n_test = x.shape[0]
        prediction = self.model.predict(x)

        norm = 1
        if self.normalizer is not None:
            try:
                norm = self.normalizer.score(x) + self.beta
            except Exception:
                pass

        if significance:
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            err_dist *= norm

            intervals = np.zeros((x.shape[0], 2))
            intervals[:, 0] = prediction - err_dist[0, :]
            intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], self.err_func.horizon_length, 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.vstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, :, 0, i] = prediction - err_dist[0, :]
                intervals[:, :, 1, i] = prediction + err_dist[0, :]

            return intervals
