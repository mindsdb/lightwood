import math
from typing import List, Union

import torch
import numpy as np
import pandas as pd
from torch.types import Number
from type_infer.dtype import dtype

from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.helpers.general import is_none


class NumericEncoder(BaseEncoder):
    """
    The numeric encoder takes numbers (float or integer) and converts it into tensors of the form:
    ``[0 if the number is none, otherwise 1, 1 if the number is positive, otherwise 0, natural_log(abs(number)), number/absolute_mean]``

    This representation is: ``[1 if the number is positive, otherwise 0, natural_log(abs(number)), number/absolute_mean]]`` if encoding target values, since target values can't be none.

    The ``absolute_mean`` is computed in the ``prepare`` method and is just the mean of the absolute values of all numbers feed to prepare (which are not none)

    ``none`` stands for any number that is an actual python ``None`` value or any sort of non-numeric value (a string, nan, inf)
    """ # noqa

    def __init__(self, data_type: dtype = None, is_target: bool = False, positive_domain: bool = False):
        """
        :param data_type: The data type of the number (integer, float, quantity)
        :param is_target: Indicates whether the encoder refers to a target column or feature column (True==target)
        :param positive_domain: Forces the encoder to always output positive values
        """
        super().__init__(is_target)
        self._abs_mean = None
        self.positive_domain = positive_domain
        self.decode_log = False
        self.output_size = 4 if not self.is_target else 3

    def prepare(self, priming_data: pd.Series):
        """
        "NumericalEncoder" uses a rule-based form to prepare results on training (priming) data. The averages etc. are taken from this distribution.

        :param priming_data: an iterable data structure containing numbers numbers which will be used to compute the values used for normalizing the encoded representations
        """ # noqa
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self._abs_mean = priming_data.abs().mean()
        self.is_prepared = True

    def encode(self, data: pd.Series):
        """
        :param data: A pandas series containing the numbers to be encoded
        :returns: A torch tensor with the representations of each number
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        # todo: wrap with try/except to cover non-real edge cases
        if not self.positive_domain:
            sign = np.vectorize(lambda x: 0 if x < 0 else 1)(data)
        else:
            sign = np.zeros(len(data))
        log_value = np.vectorize(lambda x: math.log(abs(x)) if abs(x) > 0 else -20)(data)
        log_value = np.nan_to_num(log_value, nan=0, posinf=20, neginf=-20)

        exp = np.vectorize(lambda x: x / self._abs_mean)(data)
        exp = np.nan_to_num(exp, nan=0, posinf=20, neginf=-20)

        if self.is_target:
            components = [sign, log_value, exp]
        else:
            # todo: if can't encode return 0s and log.error(f'Can\'t encode input value: {real}, exception: {e}')
            nones = np.vectorize(lambda x: 1 if is_none(x) else 0)(data)
            components = [sign, log_value, exp, nones]

        ret = torch.Tensor(np.asarray(components)).T
        return torch.Tensor(ret)

    def decode(self, encoded_values: Union[List[Number], torch.Tensor], decode_log: bool = None) -> list:
        """
        :param encoded_values: The encoded values to decode into single numbers
        :param decode_log: Whether to decode the ``log`` or ``linear`` part of the representation, since the encoded vector contains both a log and a linear part

        :returns: The decoded number
        """ # noqa
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        ret = []
        if isinstance(encoded_values, torch.Tensor):
            encoded_values = encoded_values.tolist()

        for vector in encoded_values:
            # check for none
            if len(vector) == 4 and vector[-1] == 1:
                ret.append(None)
                continue

            # edge case: divergence
            elif np.isnan(vector[0]) or vector[0] == float('inf') or \
                    np.isnan(vector[1]) or vector[1] == float('inf') or \
                    np.isnan(vector[2]) or vector[2] == float('inf'):

                log.error(f'Got weird target value to decode: {vector}')
                real_value = pow(10, 63)

            elif decode_log:
                sign = -1 if vector[0] < 0.5 else 1
                try:
                    real_value = math.exp(vector[1]) * sign
                except OverflowError:
                    real_value = pow(10, 63) * sign
            else:
                real_value = vector[2] * self._abs_mean

            if self.positive_domain:
                real_value = abs(real_value)

            # if isinstance(real_value, torch.Tensor):
            #     real_value = real_value.item()
            ret.append(real_value)
        return ret
