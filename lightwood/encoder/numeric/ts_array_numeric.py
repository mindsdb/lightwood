from typing import List, Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from lightwood.encoder import BaseEncoder
from lightwood.encoder.numeric import TsNumericEncoder


class TsArrayNumericEncoder(BaseEncoder):
    def __init__(self, timesteps: int, is_target: bool = False, positive_domain: bool = False, grouped_by=None):
        """
        This encoder handles arrays of numerical time series data by wrapping the numerical encoder with behavior specific to time series tasks.
        
        :param timesteps: length of forecasting horizon, as defined by TimeseriesSettings.window.
        :param is_target: whether this encoder corresponds to the target column.
        :param positive_domain: whether the column domain is expected to be positive numbers.
        :param grouped_by: what columns, if any, are considered to group the original column and yield multiple time series.
        """  # noqa
        super(TsArrayNumericEncoder, self).__init__(is_target=is_target)
        self.normalizers = None
        self.group_combinations = None
        self.dependencies = grouped_by
        self.data_window = timesteps
        self.positive_domain = positive_domain
        self.sub_encoder = TsNumericEncoder(is_target=is_target, positive_domain=positive_domain, grouped_by=grouped_by)
        self.output_size = self.data_window * self.sub_encoder.output_size

    def prepare(self, priming_data):
        """
        This method prepares the underlying time series numerical encoder.
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.sub_encoder.prepare(priming_data)
        self.is_prepared = True

    def encode(self, data: Iterable[Iterable], dependency_data: Optional[Dict[str, str]] = {}) -> torch.Tensor:
        """
        Encodes a list of time series arrays using the underlying time series numerical encoder.
        
        :param data: list of numerical values to encode. Its length is determined by the tss.window parameter, and all data points belong to the same time series.
        :param dependency_data: dict with values of each group_by column for the time series, used to retrieve the correct normalizer.
        
        :return: list of encoded time series arrays. Tensor is (len(data), N x K)-shaped, where N: self.data_window and K: sub-encoder # of output features.
        """  # noqa
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if self.sub_encoder.normalizers is None and self.normalizers is not None:
            self.sub_encoder.normalizers = self.normalizers
        if not dependency_data:
            dependency_data = {'__default': [None] * len(data)}

        ret = []
        for series in data:
            ret.append(self.encode_one(series, dependency_data=dependency_data))

        return torch.vstack(ret)

    def encode_one(self, data: Iterable, dependency_data: Optional[Dict[str, str]] = {}) -> torch.Tensor:
        """
        Encodes a single windowed slice of any given time series.

        :param data: windowed slice of a numerical time series.
        :param dependency_data: used to determine the correct normalizer for the input.
        
        :return: an encoded time series array, as per the underlying `TsNumericEncoder` object. 
        The output of this encoder for all time steps is concatenated, so the final shape of the tensor is (1, NxK), where N: self.data_window and K: sub-encoder # of output features. 
        """  # noqa
        ret = []

        for data_point in data:
            ret.append(self.sub_encoder.encode([data_point], dependency_data=dependency_data))

        ret = torch.hstack(ret)
        padding_size = self.output_size - ret.shape[-1]

        if padding_size > 0:
            ret = F.pad(ret, (0, padding_size))

        return ret

    def decode(self, encoded_values, dependency_data=None) -> List[List]:
        """
        Decodes a list of encoded arrays into values in their original domains.

        :param encoded_values: encoded slices of numerical time series.
        :param dependency_data: used to determine the correct normalizer for the input.

        :return: a list of decoded time series arrays.
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        encoded_values = encoded_values.reshape(encoded_values.shape[0],
                                                self.data_window,
                                                self.sub_encoder.output_size)

        ret = []
        for tensor in torch.split(encoded_values, 1, dim=0):
            ret.append(self.decode_one(tensor, dependency_data=dependency_data))

        return ret

    def decode_one(self, encoded_value, dependency_data={}) -> List:
        """
        Decodes a single window of a time series into its original domain.

        :param encoded_value: encoded slice of a numerical time series.
        :param dependency_data: used to determine the correct normalizer for the input.

        :return: a list of length TimeseriesSettings.window with decoded values for the forecasted time series.
        """
        ret = []
        for encoded_timestep in torch.split(encoded_value, 1, dim=1):
            ret.extend(self.sub_encoder.decode(encoded_timestep.squeeze(1), dependency_data=dependency_data))
        return ret
