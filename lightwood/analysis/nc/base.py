# Original author: Henrik Linusson (github.com/donlnz)
import abc
from typing import Dict
import numpy as np
from sklearn.base import BaseEstimator


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


class BaseMixerAdapter(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(BaseMixerAdapter, self).__init__()

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


class ClassifierAdapter(BaseMixerAdapter):
    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(ClassifierAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x: np.array) -> np.array:
        return self.model.predict_proba(x)


class RegressorAdapter(BaseMixerAdapter):
    def __init__(self, model: object, fit_params: Dict[str, object] = None) -> None:
        super(RegressorAdapter, self).__init__(model, fit_params)

    def _underlying_predict(self, x: np.array) -> np.array:
        return self.model.predict(x)
