import numpy as np
import pandas as pd
from typing import List
import torch
from lightwood.encoder.base import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """
    The identity encoder which does not change the data during encoding.

    Due to the restrictions of torch.Tensor, the encoding function only accepts types:

    - (nested) lists of numbers
    - np.ndarray of numbers, or
    - torch.Tensor

    Nonetypes are automatically converted into nans. This cannot be decoded.
    If self.handle_nan is True, then all nans are converted to zeros and not decoded back.

    The decode function takes in a torch.Tensor and converts it to a list of numbers.
    """  # noqa

    def __init__(self, is_target=False, handle_nan=True) -> None:
        super().__init__(is_target)
        self.handle_nan = handle_nan

    # Not all encoders need to be prepared
    def prepare(self, priming_data: pd.Series) -> None:
        self._prepared = True

    def encode(self, column_data: object) -> torch.Tensor:
        if isinstance(column_data, torch.Tensor):
            res = column_data
        else:
            res = np.array(column_data, dtype=float)  # convert None to nan
            res = torch.Tensor(res)
        if self.handle_nan:
            res = torch.nan_to_num(res)
        return res

    def decode(self, encoded_data: torch.Tensor) -> List[object]:
        return encoded_data.tolist()
