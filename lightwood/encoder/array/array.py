import torch
import pandas as pd
import numpy as np
from lightwood.encoder.base import BaseEncoder
from type_infer.dtype import dtype
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
from lightwood.helpers.general import is_none
from typing import List, Iterable


class ArrayEncoder(BaseEncoder):
    """
    Fits a normalizer for array data.

    To encode, `ArrayEncoder` returns a normalized window of previous data.
    It can be used for generic arrays, as well as for handling historical target values in time series tasks.

    Currently supported normalizing strategies are minmax for numerical arrays, and a simple one-hot for categorical arrays. See `lightwood.encoder.helpers` for more details on each approach.

    """  # noqa

    is_trainable_encoder: bool = True

    def __init__(self, stop_after: float, window: int = None, is_target: bool = False, original_type: dtype = None):
        """
        :param stop_after: time budget in seconds.
        :param window: expected length of array data.
        :param original_type: element-wise data type
        """  # noqa

        super().__init__(is_target)
        self.stop_after = stop_after
        self.original_type = original_type
        self._normalizer = None
        if window is not None:
            self.output_size = window + 1
        else:
            self.output_size = None

    def _pad_and_strip(self, array: List[object]):
        if len(array) < self.output_size:
            array = array + [0] * (self.output_size - len(array))
        if len(array) > self.output_size:
            array = array[:self.output_size]
        return array

    def prepare(self, train_priming_data: Iterable[Iterable], dev_priming_data: Iterable[Iterable]):
        """
        Prepare the array encoder for sequence data.
        :param train_priming_data: Training data of sequences
        :param dev_priming_data: Dev data of sequences
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        priming_data = pd.concat([train_priming_data, dev_priming_data])
        priming_data = priming_data.values

        if self.output_size is None:
            self.output_size = np.max([len(x) for x in priming_data if x is not None])
        for i in range(len(priming_data)):
            if is_none(priming_data[i]):
                priming_data[i] = [0] * self.output_size

        if self.original_type in (dtype.categorical, dtype.binary, dtype.cat_array, dtype.cat_tsarray):
            self._normalizer = CatNormalizer(encoder_class='ordinal')  # maybe turn into OHE encoder?
        else:
            self._normalizer = MinMaxNormalizer()  # maybe turn into numerical encoder?

        if isinstance(priming_data, pd.Series):
            priming_data = priming_data.values

        priming_data = [self._pad_and_strip(list(x)) for x in priming_data]

        self._normalizer.prepare(priming_data)
        self.output_size *= self._normalizer.output_size
        self.is_prepared = True

    def encode(self, column_data: Iterable[Iterable]) -> torch.Tensor:
        """
        Encode the properties of a sequence-of-sequence representation

        :param column_data: Input column data to be encoded
        :returns: a torch-tensor representing the encoded sequence
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(column_data, pd.Series):
            column_data = column_data.values

        for i in range(len(column_data)):
            if is_none(column_data[i]):
                column_data[i] = [0] * self.output_size
        column_data = np.array([self._pad_and_strip(list(x)) for x in column_data])

        data = torch.cat([self._normalizer.encode(column_data)], dim=-1)
        data[torch.isnan(data)] = 0.0
        data[torch.isinf(data)] = 0.0

        return data

    def decode(self, data: torch.Tensor) -> List[Iterable]:
        """
        Converts data as a list of arrays.

        :param data: Encoded data prepared by this array encoder
        :returns: A list of iterable sequences in the original data space
        """
        decoded = self._normalizer.decode(data.tolist())
        return decoded


class CatArrayEncoder(ArrayEncoder):
    def __init__(self, stop_after: float, window: int = None, is_target: bool = False):
        super(CatArrayEncoder, self).__init__(stop_after, window, is_target, original_type=dtype.cat_array)

    def prepare(self, train_priming_data: Iterable[Iterable], dev_priming_data: Iterable[Iterable]):
        super().prepare(train_priming_data, dev_priming_data)
        self.index_weights = torch.ones(size=(self.output_size,))

    def decode(self, data: torch.Tensor) -> List[Iterable]:
        data = torch.round(data)  # improves accuracy as by default ordinal encoder will truncate
        decoded = self._normalizer.decode(data.reshape(-1, 1).tolist()).reshape(1, -1)
        return decoded


class NumArrayEncoder(ArrayEncoder):
    def __init__(self, stop_after: float, window: int = None, is_target: bool = False, positive_domain: bool = False):
        self.positive_domain = positive_domain
        super(NumArrayEncoder, self).__init__(stop_after, window, is_target, original_type=dtype.num_array)
