import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class PlattCalibrator(BaseEstimator):
    """
    Original source: https://github.com/ethen8181/machine-learning/, MIT Licensed. Retrieved: 20-12-2021.
    Boils down to applying a Logistic Regression.
    Parameters
    ----------
    log_odds : bool, default True
        Logistic Regression assumes a linear relationship between its input
        and the log-odds of the class probabilities. Converting the probability
        to log-odds scale typically improves performance.
    Attributes
    ----------
    coef_ : ndarray of shape (1,)
        Binary logistic regression's coefficient.
    intercept_ : ndarray of shape (1,)
        Binary logistic regression's intercept.
    """

    def __init__(self, log_odds: bool = True):
        self.log_odds = log_odds

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        y_true : 1d ndarray
            Binary true targets.
        Returns
        -------
        self
        """
        self.fit_predict(y_prob, y_true)
        return self

    @staticmethod
    def _convert_to_log_odds(y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob = np.log(y_prob / (1 - y_prob))
        return y_prob

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        output = self._transform(y_prob)
        return output

    def _transform(self, y_prob: np.ndarray) -> np.ndarray:
        output = y_prob * self.coef_[0] + self.intercept_
        output = 1 / (1 + np.exp(-output))
        return output

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Chain the .fit and .predict step together.
        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.
        y_true : 1d ndarray
            Binary true targets.
        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        # the class expects 2d ndarray as input features
        logistic = LogisticRegression(C=1e10, solver='lbfgs')
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        self.coef_ = logistic.coef_[0]
        self.intercept_ = logistic.intercept_

        y_calibrated_prob = self._transform(y_prob)
        return y_calibrated_prob
