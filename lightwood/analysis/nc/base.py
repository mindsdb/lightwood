# Original author: Henrik Linusson (github.com/donlnz)
import abc
from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator


from lightwood.analysis.nc.util import t_softmax


class RegressorMixin(object):
    def __init__(self) -> None:
        super(RegressorMixin, self).__init__()

    @classmethod
    def get_problem_type(cls):
        return 'regression'


class ClassifierMixin(object):
    def __init__(self) -> None:
        super(ClassifierMixin, self).__init__()

    @classmethod
    def get_problem_type(cls) -> str:
        return 'classification'


class TSMixin(object):
    def __init__(self) -> None:
        super(TSMixin, self).__init__()

    @classmethod
    def get_problem_type(cls):
        return 'time-series'


class BaseModelAdapter(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(BaseModelAdapter, self).__init__()

        self.model = model
        self.last_x, self.last_y = None, None
        self.clean = False
        self.fit_params = {} if fit_params is None else fit_params

    def fit(self, x: np.array, y: np.array) -> None:
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

    def predict(self, x: np.array) -> np.array:
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
    def _underlying_predict(self, x: np.array) -> np.array:
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
    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(ClassifierAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x: np.array) -> np.array:
        return self.model.predict_proba(x)


class RegressorAdapter(BaseModelAdapter):
    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(RegressorAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x: np.array) -> np.array:
        return self.model.predict(x)


class TSAdapter(BaseModelAdapter):
    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(TSAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x: np.array) -> np.array:
        return self.model.predict(x)


class CachedRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(CachedRegressorAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        """ At this point, the predictor has already been trained, but this
        has to be called to setup some things in the nonconformist backend """
        pass

    def predict(self, x=None):
        """ Same as in .fit()
        :return: np.array (n_test, n_classes) with class probability estimates """
        return self.prediction_cache


class CachedClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(CachedClassifierAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None
        self.tempscale = True

    def fit(self, x=None, y=None):
        """ At this point, the predictor has already been trained, but this
        has to be called to setup some things in the nonconformist backend """
        pass

    def predict(self, x=None):
        """ Same as in .fit()
        :return: np.array (n_test, n_classes) with class probability estimates """
        if self.tempscale:
            return t_softmax(self.prediction_cache, t=0.5)
        else:
            return self.prediction_cache


class CachedTSAdapter(TSAdapter):
    def __init__(self, model, fit_params=None):
        super(CachedTSAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        pass

    def predict(self, x=None):
        return self.prediction_cache
