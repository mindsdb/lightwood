import numpy as np
from typing import List, Iterable, Union
import torch
from lightwood.encoder.base import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """
    Preserves the original input data without any transformations.

    Due to the restrictions of torch.Tensor, the encoding function only accepts types:

    - (nested) lists of numbers
    - np.ndarray of numbers, or
    - torch.Tensor

    Nonetypes are automatically converted into nans. This cannot be decoded.
    If self.handle_nan is True, then all nans are converted to zeros and not decoded back.

    The decode function takes in a torch.Tensor and converts it to a list of numbers.
    """  # noqa

    def __init__(self, is_target: bool = False, handle_nan: bool = True) -> None:
        """
        :param is_target: Whether encoder represents the target column
        :param handle_nan: If True, converts any NaN values to 0 via `torch.nan_to_num`.
        """
        super().__init__(is_target)
        self.handle_nan = handle_nan

    # Not all encoders need to be prepared
    def prepare(self, priming_data: Iterable[Union[float, int]]) -> None:
        """ Nothing to prepare """
        self._prepared = True

    def encode(self, column_data: Iterable[Union[float, int]]) -> torch.Tensor:
        """
        Converts input data into a torch tensor with NaN handling, if necessary.

        :param column_data: Input iterable of numerical data
        :returns: Torch tensor of the data type
        """
        if isinstance(column_data, torch.Tensor):
            res = column_data
        else:
            res = np.array(column_data, dtype=float)  # converts None to nan
            res = torch.Tensor(res)

        if self.handle_nan:
            res = torch.nan_to_num(res)
        return res

    def decode(self, encoded_data: torch.Tensor) -> List[object]:
        """ Returns a list of the input data """
        return encoded_data.tolist()
