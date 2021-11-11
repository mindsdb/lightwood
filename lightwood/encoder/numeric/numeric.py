import math
from typing import Iterable, List, Union
import torch
import numpy as np
from torch.types import Number
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.helpers.general import is_none
from lightwood.api.dtype import dtype


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
        self._type = data_type
        self._abs_mean = None
        self.positive_domain = positive_domain
        self.decode_log = False
        self.output_size = 4 if not self.is_target else 3

    def prepare(self, priming_data: Iterable):
        """
        "NumericalEncoder" uses a rule-based form to prepare results on training (priming) data. The averages etc. are taken from this distribution.

        :param priming_data: an iterable data structure containing numbers numbers which will be used to compute the values used for normalizing the encoded representations
        """ # noqa
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        value_type = 'int'
        for number in priming_data:
            if not is_none(number):
                if int(number) != number:
                    value_type = 'float'

        self._type = value_type if self._type is None else self._type
        non_null_priming_data = [x for x in priming_data if not is_none(x)]
        self._abs_mean = np.mean(np.abs(non_null_priming_data))
        self.is_prepared = True

    def encode(self, data: Iterable):
        """
        :param data: An iterable data structure containing the numbers to be encoded

        :returns: A torch tensor with the representations of each number
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        for real in data:
            try:
                real = float(real)
            except Exception:
                real = None
            if self.is_target:
                # Will crash if ``real`` is not a float, this is fine, targets should always have a value
                vector = [0] * 3
                vector[0] = 1 if real < 0 and not self.positive_domain else 0
                vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                vector[2] = real / self._abs_mean

            else:
                vector = [0] * 4
                try:
                    if is_none(real):
                        vector[0] = 0
                    else:
                        vector[0] = 1
                        vector[1] = math.log(abs(real)) if abs(real) > 0 else -20
                        vector[2] = 1 if real < 0 and not self.positive_domain else 0
                        vector[3] = real / self._abs_mean
                except Exception as e:
                    vector = [0] * 4
                    log.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(vector)

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
            if self.is_target:
                if np.isnan(
                        vector[0]) or vector[0] == float('inf') or np.isnan(
                        vector[1]) or vector[1] == float('inf') or np.isnan(
                        vector[2]) or vector[2] == float('inf'):
                    log.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10, 63)
                else:
                    if decode_log:
                        sign = -1 if vector[0] > 0.5 else 1
                        try:
                            real_value = math.exp(vector[1]) * sign
                        except OverflowError:
                            real_value = pow(10, 63) * sign
                    else:
                        real_value = vector[2] * self._abs_mean

                    if self.positive_domain:
                        real_value = abs(real_value)

                    if self._type == 'int':
                        real_value = int(real_value)

            else:
                if vector[0] < 0.5:
                    ret.append(None)
                    continue

                real_value = vector[3] * self._abs_mean

                if self._type == 'int':
                    real_value = round(real_value)

            if isinstance(real_value, torch.Tensor):
                real_value = real_value.item()
            ret.append(real_value)
        return ret
