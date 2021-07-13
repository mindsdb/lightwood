# Original author: Henrik Linusson (github.com/donlnz)

import abc
import numpy as np

from sklearn.base import BaseEstimator


class RegressorMixin(object):
    def __init__(self):
        super(RegressorMixin, self).__init__()

    @classmethod
    def get_problem_type(cls):
        return 'regression'


class ClassifierMixin(object):
    def __init__(self):
        super(ClassifierMixin, self).__init__()

    @classmethod
    def get_problem_type(cls):
        return 'classification'


class BaseModelAdapter(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, fit_params=None):
        super(BaseModelAdapter, self).__init__()

        self.model = model
        self.last_x, self.last_y = None, None
        self.clean = False
        self.fit_params = {} if fit_params is None else fit_params

    def fit(self, x, y):
        """Fits the model.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the model.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the model.

        Returns
        -------
        None
        """

        self.model.fit(x, y, **self.fit_params)
        self.clean = False

    def predict(self, x):
        """Returns the prediction made by the underlying model.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of test examples.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted outputs of test examples.
        """
        if (
                not self.clean or
                self.last_x is None or
                self.last_y is None or
                not np.array_equal(self.last_x, x)
        ):
            self.last_x = x
            self.last_y = self._underlying_predict(x)
            self.clean = True

        return self.last_y.copy()

    @abc.abstractmethod
    def _underlying_predict(self, x):
        """Produces a prediction using the encapsulated model.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of test examples.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted outputs of test examples.
        """
        pass


class ClassifierAdapter(BaseModelAdapter):
    def __init__(self, model, fit_params=None):
        super(ClassifierAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x):
        return self.model.predict_proba(x)


class RegressorAdapter(BaseModelAdapter):
    def __init__(self, model, fit_params=None):
        super(RegressorAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x):
        return self.model.predict(x)


class OobMixin(object):
    """ OOB: out-of-bag"""
    def __init__(self, model, fit_params=None):
        super(OobMixin, self).__init__(model, fit_params)
        self.train_x = None

    def fit(self, x, y):
        super(OobMixin, self).fit(x, y)
        self.train_x = x

    def _underlying_predict(self, x):
        # TODO: sub-sampling of ensemble for test patterns
        oob = x == self.train_x

        if hasattr(oob, 'all'):
            oob = oob.all()

        if oob:
            return self._oob_prediction()
        else:
            return super(OobMixin, self)._underlying_predict(x)


class OobClassifierAdapter(OobMixin, ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(OobClassifierAdapter, self).__init__(model, fit_params)

    def _oob_prediction(self):
        return self.model.oob_decision_function_


class OobRegressorAdapter(OobMixin, RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(OobRegressorAdapter, self).__init__(model, fit_params)

    def _oob_prediction(self):
        return self.model.oob_prediction_
