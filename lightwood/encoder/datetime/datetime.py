from typing import Union

import torch
import numpy as np
import pandas as pd

from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log


class DatetimeEncoder(BaseEncoder):
    """
    This encoder produces an encoded representation for timestamps.

    The approach consists on decomposing the timestamp objects into its constituent units (e.g. month, year, etc), and describing each of those with a single value that represents the magnitude in a sensible cycle length.
    """  # noqa
    def __init__(self, is_target: bool = False):
        super().__init__(is_target)
        self.constant_keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.constant_vals = torch.Tensor([3000.0, 12.0, 31.0, 24.0, 60.0, 60.0])  # cycle length
        self.constant_map = {k: v.item() for k, v in zip(self.constant_keys, self.constant_vals)}

        self.output_size = len(self.constant_keys)
        self.empty_vector = np.zeros((self.output_size, ))

        self.max_vals = torch.Tensor([pd.Timestamp.max.year - 1, 12, 31, 23, 59, 59])
        self.min_vals = torch.Tensor([pd.Timestamp.min.year + 1, 1, 1, 0, 0, 0])

    def prepare(self, priming_data):
        self.is_prepared = True

    def encode(self, data: Union[np.ndarray, pd.Series]) -> torch.Tensor:
        """
        :param data: a pandas series with numerical dtype, previously cleaned with dataprep_ml
        :return: encoded data, shape (len(data), self.output_size)
        """
        if type(data) not in (np.ndarray, pd.Series):
            raise Exception(f'Data should be pd.Series or np.ndarray! Got: {type(data)}')
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = data.fillna(pd.Timestamp.max.timestamp())  # TODO: replace with mean?
        ret = [pd.to_datetime(data, unit='s', origin=-1, utc=True)]
        for i, attr in enumerate(self.constant_keys):
            def _get_ts_attr(ts):
                return getattr(ts, attr)
            component = ret[0].apply(_get_ts_attr)
            component = component / self.constant_vals[i].item()
            ret.append(component)

        out = torch.Tensor(ret[1:])  # drop column with timestamp objects
        out = torch.transpose(out, 0, 1)  # swap dimensions to shape as (B, self.output_size)
        return out

    def decode(self, encoded_data: torch.Tensor, return_as_datetime=False) -> list:
        if len(encoded_data.shape) > 2 and encoded_data.shape[0] == 1:
            encoded_data = encoded_data.squeeze(0)

        rounded = torch.round(torch.multiply(encoded_data, self.constant_vals))
        high_bounded = torch.minimum(rounded, self.max_vals)
        low_bounded = torch.maximum(high_bounded, self.min_vals)
        ret = low_bounded.long()

        df = pd.DataFrame(ret, columns=self.constant_keys)
        nan_mask = df[
            (df['year'] == int(self.max_vals[0])) &
            (df['month'] == pd.Timestamp.max.month) &
            (df['day'] == pd.Timestamp.max.day)
        ].index
        dt = pd.to_datetime(df, utc=True)

        if not hasattr(dt, 'dt'):
            log.warning('DatetimeEncoder has failed to decode using microsecond precision, reverting to nanosecond. This may lead to minor discrepancies in reconstruction.')  # noqa

        if return_as_datetime is True:
            dt = dt.dt.to_pydatetime()  # return to Python datetime microsecond precision
            decoded = dt
        else:
            decoded = dt.values.astype(np.float64) // 10 ** 9

        decoded[nan_mask] = np.nan
        return decoded.tolist()
