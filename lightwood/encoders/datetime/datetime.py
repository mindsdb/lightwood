import datetime
import numpy as np
import torch
from lightwood.encoders.encoder_base import BaseEncoder


class DatetimeEncoder(BaseEncoder):
    def __init__(self, is_target=False, sinusoidal=False):
        super().__init__(is_target)
        self.sinusoidal = sinusoidal
        self.fields = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second']
        self.constants = {'year': 3000.0, 'month': 12.0, 'day': 31.0, 'weekday': 7.0,
                          'hour': 24.0, 'minute': 60.0, 'second': 60.0}

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self._prepared = True

    def encode(self, data):
        """
        Encodes a list of unix_timestamps
        :param data: list of unix_timestamps (unix_timestamp resolution is seconds)
        :return: a list of vectors
        """
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []

        for unix_timestamp in data:

            if unix_timestamp is None:
                if self.sinusoidal:
                    vector = [0, 1] * len(self.fields)
                else:
                    vector = [0] * len(self.fields)
            else:
                c = self.constants
                date = datetime.datetime.fromtimestamp(unix_timestamp)
                vector = [date.year / c['year'], date.month / c['month'], date.day / c['day'],
                          date.weekday() / c['weekday'], date.hour / c['hour'],
                          date.minute / c['minute'], date.second / c['second']]
                if self.sinusoidal:
                    vector = np.array([(np.sin(n), np.cos(n)) for n in vector]).flatten()

            ret.append(vector)

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_data, return_as_datetime=False):
        ret = []
        for vector in encoded_data.tolist():

            if sum(vector) == 0:
                ret.append(None)

            else:
                if self.sinusoidal:
                    vector = list(map(lambda x: np.arcsin(x), vector))[::2]
                c = self.constants
                dt = datetime.datetime(year=round(vector[0] * c['year']), month=round(vector[1] * c['month']),
                                       day=round(vector[2] * c['day']), hour=round(vector[4] * c['hour']),
                                       minute=round(vector[5] * c['minute']), second=round(vector[6] * c['second']))

                if return_as_datetime is True:
                    ret.append(dt)
                else:
                    ret.append(
                        round(dt.timestamp())
                    )

        return ret
