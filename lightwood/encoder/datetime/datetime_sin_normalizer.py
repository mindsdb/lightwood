import datetime
import calendar
import numpy as np
import pandas as pd  # @TODO: remove?
import torch
from lightwood.encoder.base import BaseEncoder
from collections.abc import Iterable
from lightwood.helpers.general import is_none


class DatetimeNormalizerEncoder(BaseEncoder):
    def __init__(self, is_target: bool = False, sinusoidal: bool = False):
        super().__init__(is_target)
        self.sinusoidal = sinusoidal
        self.fields = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
        self.constants = {'year': 3000.0, 'month': 12.0, 'weekday': 7.0,
                          'hour': 24.0, 'minute': 60.0, 'second': 60.0}
        if self.sinusoidal:
            self.output_size = 2
        else:
            self.output_size = 7

    def prepare(self, priming_data):
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.is_prepared = True

    def encode(self, data):
        """
        :param data: # @TODO: receive a consistent data type here; currently either list of lists or pd.Series w/lists
        :return: encoded data
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(data, pd.Series):
            data = data.values
        if not isinstance(data[0], Iterable):
            data = [data]

        ret = [self.encode_one(row) for row in data]

        return torch.Tensor(ret)

    def encode_one(self, data):
        """
        Encodes a list of unix_timestamps, or a list of tensors with unix_timestamps
        :param data: list of unix_timestamps (unix_timestamp resolution is seconds)
        :return: a list of vectors
        """
        ret = []
        for unix_timestamp in data:
            if is_none(unix_timestamp):
                if self.sinusoidal:
                    vector = [0, 1] * len(self.fields)
                else:
                    vector = [0] * len(self.fields)
            else:
                c = self.constants
                if isinstance(unix_timestamp, torch.Tensor):
                    unix_timestamp = unix_timestamp.item()
                date = datetime.datetime.fromtimestamp(unix_timestamp)
                day_constant = calendar.monthrange(date.year, date.month)[1]
                vector = [date.year / c['year'], date.month / c['month'], date.day / day_constant,
                          date.weekday() / c['weekday'], date.hour / c['hour'],
                          date.minute / c['minute'], date.second / c['second']]
                if self.sinusoidal:
                    vector = np.array([(np.sin(n), np.cos(n)) for n in vector]).flatten()

            ret.append(vector)

        return ret

    def decode(self, encoded_data, return_as_datetime=False):
        ret = []
        if len(encoded_data.shape) > 2 and encoded_data.shape[0] == 1:
            encoded_data = encoded_data.squeeze(0)

        for vector in encoded_data.tolist():
            ret.append(self.decode_one(vector, return_as_datetime=return_as_datetime))

        return ret

    def decode_one(self, vector, return_as_datetime=False):
        if sum(vector) == 0:
            decoded = None

        else:
            if self.sinusoidal:
                vector = list(map(lambda x: np.arcsin(x), vector))[::2]
            c = self.constants

            year = max(0, round(vector[0] * c['year']))
            month = max(1, min(12, round(vector[1] * c['month'])))
            day_constant = calendar.monthrange(year, month)[-1]
            day = max(1, min(round(vector[2] * day_constant), day_constant))
            hour = max(0, min(23, round(vector[4] * c['hour'])))
            minute = max(0, min(59, round(vector[5] * c['minute'])))
            second = max(0, min(59, round(vector[6] * c['second'])))

            dt = datetime.datetime(year=year, month=month, day=day, hour=hour,
                                   minute=minute, second=second)

            if return_as_datetime is True:
                decoded = dt
            else:
                decoded = round(dt.timestamp())

        return decoded
