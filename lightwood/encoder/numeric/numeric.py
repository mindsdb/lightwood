import math
from typing import Union, Dict
from copy import deepcopy as dc

import torch
import numpy as np
import pandas as pd
from type_infer.dtype import dtype

from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.general import is_none


class NumericEncoder(BaseEncoder):
    """
    The numeric encoder takes numbers (float or integer) and converts it into tensors of the form:
    ``[0 if the number is none, otherwise 1, 1 if the number is positive, otherwise 0, natural_log(abs(number)), number/absolute_mean]``

    This representation is: ``[1 if the number is positive, otherwise 0, natural_log(abs(number)), number/absolute_mean]]`` if encoding target values, since target values can't be none.

    The ``absolute_mean`` is computed in the ``prepare`` method and is just the mean of the absolute values of all numbers feed to prepare (which are not none)

    ``none`` stands for any number that is an actual python ``None`` value or any sort of non-numeric value (a string, nan, inf)
    """  # noqa

    def __init__(self, data_type: dtype = None,
                 target_weights: Dict[float, float] = None,
                 is_target: bool = False,
                 positive_domain: bool = False):
        """
        :param data_type: The data type of the number (integer, float, quantity)
        :param target_weights: a dictionary of weights to use on the examples.
        :param is_target: Indicates whether the encoder refers to a target column or feature column (True==target)
        :param positive_domain: Forces the encoder to always output positive values
        """
        super().__init__(is_target)
        self._abs_mean = None
        self.positive_domain = positive_domain
        self.decode_log = False
        self.output_size = 4 if not self.is_target else 3

        # Weight-balance info if encoder represents target
        self.target_weights = None
        self.index_weights = None
        if self.is_target and target_weights is not None:
            self.target_weights = dc(target_weights)
            self.index_weights = torch.tensor(list(self.target_weights.values()))

    def prepare(self, priming_data: pd.Series):
        """
        "NumericalEncoder" uses a rule-based form to prepare results on training (priming) data. The averages etc. are taken from this distribution.

        :param priming_data: an iterable data structure containing numbers numbers which will be used to compute the values used for normalizing the encoded representations
        """  # noqa
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self._abs_mean = priming_data.abs().mean()
        self.is_prepared = True

    def encode(self, data: Union[np.ndarray, pd.Series]):
        """
        :param data: A pandas series or numpy array containing the numbers to be encoded
        :returns: A torch tensor with the representations of each number
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(data, pd.Series):
            data = data.values

        inp_data = np.nan_to_num(data.astype(float), nan=0, posinf=np.finfo(np.float32).max,
                                 neginf=np.finfo(np.float32).min)  # noqa
        if not self.positive_domain:
            sign = np.vectorize(self._sign_fn, otypes=[float])(inp_data)
        else:
            sign = np.zeros(len(data))
        log_value = np.vectorize(self._log_fn, otypes=[float])(inp_data)
        log_value = np.nan_to_num(log_value, nan=0, posinf=20, neginf=-20)

        norm = np.vectorize(self._norm_fn, otypes=[float])(inp_data)
        norm = np.nan_to_num(norm, nan=0, posinf=20, neginf=-20)

        if self.is_target:
            components = [sign, log_value, norm]
        else:
            nones = np.vectorize(self._none_fn, otypes=[float])(data)
            components = [sign, log_value, norm, nones]

        return torch.Tensor(np.asarray(components)).T

    @staticmethod
    def _sign_fn(x: float) -> float:
        return 0 if x < 0 else 1

    @staticmethod
    def _log_fn(x: float) -> float:
        return math.log(abs(x)) if abs(x) > 0 else -20

    def _norm_fn(self, x: float) -> float:
        return x / self._abs_mean

    @staticmethod
    def _none_fn(x: float) -> float:
        return 1 if is_none(x) else 0

    def decode(self, encoded_values: torch.Tensor, decode_log: bool = None) -> list:
        """
        :param encoded_values: The encoded values to decode into single numbers
        :param decode_log: Whether to decode the ``log`` or ``linear`` part of the representation, since the encoded vector contains both a log and a linear part

        :returns: The decoded array
        """  # noqa

        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        # force = True prevents side effects on the original encoded_values
        ev = encoded_values.numpy(force=True)

        # set "divergent" value as default (note: finfo.max() instead of pow(10, 63))
        ret = np.full((ev.shape[0],), dtype=float, fill_value=np.finfo(np.float64).max)

        # `none` filter (if not a target column)
        if not self.is_target:
            mask_none = ev[:, -1] == 1
            ret[mask_none] = np.nan
        else:
            mask_none = np.zeros_like(ret)

        # sign component
        sign = np.ones(ev.shape[0], dtype=float)
        mask_sign = ev[:, 0] < 0.5
        sign[mask_sign] = -1

        # real component
        if decode_log:
            real_value = np.exp(ev[:, 1]) * sign
            overflow_mask = ev[:, 1] >= 63
            real_value[overflow_mask] = 10 ** 63
            valid_mask = ~overflow_mask
        else:
            real_value = ev[:, 2] * self._abs_mean
            valid_mask = np.ones_like(real_value, dtype=bool)

        # final filters
        if self.positive_domain:
            real_value = abs(real_value)

        ret[valid_mask] = real_value[valid_mask]

        # set nan back to None
        if mask_none.sum() > 0:
            ret = ret.astype(object)
            ret[mask_none] = None

        return ret.tolist()  # TODO: update signature on BaseEncoder and replace all encs to return ndarrays

    def get_weights(self, label_data):
        # get a sorted list of intervals to assign weights. Keys are the interval edges.
        target_weight_keys = np.array(list(self.target_weights.keys()))
        target_weight_values = np.array(list(self.target_weights.values()))
        sorted_indices = np.argsort(target_weight_keys)

        # get sorted arrays for vector numpy operations
        target_weight_keys = target_weight_keys[sorted_indices]
        target_weight_values = target_weight_values[sorted_indices]

        # find the indices of the bins according to the keys. clip to the length of the weight values (search sorted
        # returns indices from 0 to N with N = len(target_weight_keys).
        assigned_target_weight_indices = np.clip(a=np.searchsorted(target_weight_keys, label_data),
                                                 a_min=0,
                                                 a_max=len(target_weight_keys) - 1).astype(np.int32)

        return target_weight_values[assigned_target_weight_indices]

