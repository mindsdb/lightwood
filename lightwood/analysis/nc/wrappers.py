import os
import torch
import numpy as np
from scipy.interpolate import interp1d
from torch.nn.functional import softmax
from base import RegressorAdapter
from base import ClassifierAdapter
from nc import BaseScorer, RegressionErrFunc

from lightwood.api.predictor import Predictor
from mindsdb_native.config import CONFIG


def t_softmax(x, t=1.0, axis=1):
	""" Softmax with temperature scaling """
	return softmax(torch.Tensor(x) / t, dim=axis).numpy()


def clear_icp_state(icp):
	""" We clear last_x and last_y to minimize file size. Model has to be cleared because it cannot be pickled. """
	icp.model.model = None
	icp.model.last_x = None
	icp.model.last_y = None
	if icp.normalizer is not None:
	    icp.normalizer.model = None


def restore_icp_state(col, hmd, session):
	icps = hmd['icp'][col]
	try:
	    predictor = session.transaction.model_backend.predictor
	except AttributeError:
	    model_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, hmd['name'], 'lightwood_data')
	    predictor = Predictor(load_from_path=model_path)

	for group, icp in icps.items():
	    if group not in ['__mdb_groups', '__mdb_group_keys']:
	        icp.nc_function.model.model = predictor

	# restore model in normalizer
	for group, icp in icps.items():
	    if group not in ['__mdb_groups', '__mdb_group_keys']:
	        if icp.nc_function.normalizer is not None:
	            icp.nc_function.normalizer.model = icp.nc_function.model.model


class BoostedAbsErrorErrFunc(RegressionErrFunc):
	""" Calculates absolute error nonconformity for regression problems. Applies linear interpolation
	for nonconformity scores when we have less than 100 samples in the validation dataset.
	"""
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
	        nc = interp(np.linspace(0, nc.size-1, 100))
	    border = min(max(border, 0), nc.size - 1)
	    return np.vstack([nc[border], nc[border]])


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


class SelfawareNormalizer(BaseScorer):
	def __init__(self, fit_params=None):
	    super(SelfawareNormalizer, self).__init__()
	    self.prediction_cache = None
	    self.output_column = fit_params['output_column']

	def fit(self, x, y):
	    """ No fitting is needed, as the self-aware model is trained in Lightwood """
	    pass

	def score(self, true_input, y=None):
	    sa_score = self.prediction_cache

	    if sa_score is None:
	        sa_score = np.ones(true_input.shape[0])  # by default, normalizing factor is 1 for all predictions
	    else:
	        sa_score = np.array(sa_score)

	    return sa_score
