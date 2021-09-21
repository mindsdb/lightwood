from lightwood.analysis.nc.base import RegressorAdapter, ClassifierAdapter
from lightwood.analysis.nc.util import t_softmax


class ConformalRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalRegressorAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        """ At this point, the predictor has already been trained, but this
        has to be called to setup some things in the nonconformist backend """
        pass

    def predict(self, x=None):
        """ Same as in .fit()
        :return: np.array (n_test, n_classes) with class probability estimates """
        return self.prediction_cache


class ConformalClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalClassifierAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        """ At this point, the predictor has already been trained, but this
        has to be called to setup some things in the nonconformist backend """
        pass

    def predict(self, x=None):
        """ Same as in .fit()
        :return: np.array (n_test, n_classes) with class probability estimates """
        return t_softmax(self.prediction_cache, t=0.5)
