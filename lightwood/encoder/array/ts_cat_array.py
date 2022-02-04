from typing import List, Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from lightwood.encoder import BaseEncoder
from lightwood.encoder.categorical import OneHotEncoder


class TsCatArrayEncoder(BaseEncoder):
    def __init__(self, timesteps: int, is_target: bool = False, grouped_by=None):
        """
        This encoder handles arrays of categorical time series data by wrapping the OHE encoder with behavior specific to time series tasks.

        :param timesteps: length of forecasting horizon, as defined by TimeseriesSettings.window.
        :param is_target: whether this encoder corresponds to the target column.
        :param grouped_by: what columns, if any, are considered to group the original column and yield multiple time series.
        """  # noqa
        super(TsCatArrayEncoder, self).__init__(is_target=is_target)
        self.group_combinations = None
        self.dependencies = grouped_by
        self.data_window = timesteps
        self.sub_encoder = OneHotEncoder(is_target=is_target, use_unknown=False)

    def prepare(self, priming_data):
        """
        This method prepares the underlying time series numerical encoder.
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.sub_encoder.prepare(priming_data)
        self.output_size = self.data_window * self.sub_encoder.output_size
        self.rev_map = self.sub_encoder.rev_map
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

        ret = []
        for series in data:
            ret.append(self.encode_one(series))

        return torch.vstack(ret)

    def encode_one(self, data: Iterable) -> torch.Tensor:
        """
        Encodes a single windowed slice of any given time series.

        :param data: windowed slice of a numerical time series.

        :return: an encoded time series array, as per the underlying `TsNumericEncoder` object. 
        The output of this encoder for all time steps is concatenated, so the final shape of the tensor is (1, NxK), where N: self.data_window and K: sub-encoder # of output features. 
        """  # noqa
        ret = []

        for data_point in data:
            ret.append(self.sub_encoder.encode([data_point]))

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
            ret.append(self.decode_one(tensor))

        return ret

    def decode_one(self, encoded_value) -> List:
        """
        Decodes a single window of a time series into its original domain.

        :param encoded_value: encoded slice of a numerical time series.
        :param dependency_data: used to determine the correct normalizer for the input.

        :return: a list of length TimeseriesSettings.window with decoded values for the forecasted time series.
        """
        ret = []
        for encoded_timestep in torch.split(encoded_value, 1, dim=1):
            ret.extend(self.sub_encoder.decode(encoded_timestep.squeeze(1)))
        return ret
