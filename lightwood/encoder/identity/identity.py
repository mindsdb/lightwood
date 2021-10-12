from typing import List
import torch
from lightwood.encoder.base import BaseEncoder


class IdentityEncoder(BaseEncoder):

    def __init__(self, is_target=False) -> None:
        super().__init__(is_target)

    # Not all encoders need to be prepared
    def prepare(self, priming_data) -> None:
        self._prepared = True

    def encode(self, column_data) -> torch.Tensor:
        if type(column_data) == torch.Tensor:
            return column_data
        return torch.Tensor(column_data)

    def decode(self, encoded_data) -> List[object]:
        if type(encoded_data) == torch.Tensor:
            return encoded_data.tolist()
        return list(encoded_data)
