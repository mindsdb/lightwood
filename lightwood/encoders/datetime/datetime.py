import datetime
import torch
from lightwood.encoders.encoder_base import BaseEncoder


class DatetimeEncoder(BaseEncoder):
    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._prepared = True

    def encode(self, data):
        """
        Encodes a list of unix_timestamps
        :param data: list of unix_timestamps (unix_timestamp resolution is seconds)
        :return: a list of vectors
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        ret = []

        for unix_timestamp in data:

            if unix_timestamp is None:
                vector = [0] * 7
            else:
                date = datetime.datetime.fromtimestamp(unix_timestamp)
                vector = [date.year / 3000.0, date.month / 12.0, date.day / 31.0,
                          date.weekday() / 7.0, date.hour / 24.0, date.minute / 60.0, date.second / 60.0]

            ret.append(vector)

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_data, return_as_datetime=False):
        ret = []
        for vector in encoded_data.tolist():

            if sum(vector) == 0:
                ret.append(None)

            else:
                dt = datetime.datetime(year=round(vector[0] * 3000), month=round(vector[1] * 12),
                                       day=round(vector[2] * 31), hour=round(vector[4] * 24),
                                       minute=round(vector[5] * 60), second=round(vector[6] * 60))

                if return_as_datetime is True:
                    ret.append(dt)
                else:
                    ret.append(
                        round(dt.timestamp())
                    )

        return ret


if __name__ == "__main__":

    data = [1555943147, None, 1555943147]

    enc = DatetimeEncoder()
    enc.prepare_encoder([])
    print (enc.decode(enc.encode(data)))

    # print(enc.decode(enc.encode(['not there', 'time', 'tokens'])))
