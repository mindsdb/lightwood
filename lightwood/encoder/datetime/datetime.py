import datetime
import calendar
from typing import Optional
import torch
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.general import is_none


class DatetimeEncoder(BaseEncoder):
    """
    This encoder produces an encoded representation for timestamps.

    The approach consists on decomposing the timestamp objects into its constituent units (e.g. day-of-week, month, year, etc), and describing each of those with a single value that represents the magnitude in a sensible cycle length.
    """  # noqa
    def __init__(self, is_target: bool = False):
        super().__init__(is_target)
        self.fields = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
        self.constants = {'year': 3000.0, 'month': 12.0, 'weekday': 7.0,
                          'hour': 24.0, 'minute': 60.0, 'second': 60.0}
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

        ret = [self.encode_one(unix_timestamp) for unix_timestamp in data]

        return torch.Tensor(ret)

    def encode_one(self, unix_timestamp: Optional[float]):
        """
        Encodes a list of unix_timestamps, or a list of tensors with unix_timestamps
        :param data: list of unix_timestamps (unix_timestamp resolution is seconds)
        :return: a list of vectors
        """
        if is_none(unix_timestamp):
            vector = [0] * len(self.fields)
        else:
            c = self.constants
            date = datetime.datetime.fromtimestamp(unix_timestamp)
            day_constant = calendar.monthrange(date.year, date.month)[1]
            vector = [date.year / c['year'], date.month / c['month'], date.day / day_constant,
                      date.weekday() / c['weekday'], date.hour / c['hour'],
                      date.minute / c['minute'], date.second / c['second']]
        return vector

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
