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

    Nonetypes are not supported.

    The decode function takes in a torch.Tensor and converts it to a list of numbers.
    """  # noqa

    def __init__(self, is_target=False) -> None:
        super().__init__(is_target)

    # Not all encoders need to be prepared
    def prepare(self, priming_data: pd.Series) -> None:
        self._prepared = True

    def encode(self, column_data: object) -> torch.Tensor:
        if type(column_data) == torch.Tensor:
            return column_data
        return torch.Tensor(column_data)

    def decode(self, encoded_data: torch.Tensor) -> List[object]:
        return encoded_data.tolist()
