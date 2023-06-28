from typing import List, Union
from collections import defaultdict
import pandas as pd
import numpy as np
import torch

from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.constants import _UNCOMMON_WORD


class SimpleLabelEncoder(BaseEncoder):
    """
    Simple encoder that assigns a unique integer to every observed label.
    
    Allocates an `unknown` label by default to index 0.
    
    Labels must be exact matches between inference and training (e.g. no .lower() on strings is performed here).
    """  # noqa

    def __init__(self, is_target=False, normalize=True) -> None:
        super().__init__(is_target)
        self.label_map = defaultdict(int)  # UNK category maps to 0
        self.inv_label_map = {}  # invalid encoded values are mapped to None in `decode`
        self.output_size = 1
        self.n_labels = None
        self.normalize = normalize

    def prepare(self, priming_data: Union[list, pd.Series]) -> None:
        if not isinstance(priming_data, pd.Series):
            priming_data = pd.Series(priming_data)

        for i, v in enumerate(priming_data.unique()):
            if v is not None:
                self.label_map[str(v)] = int(i + 1)  # leave 0 for UNK
        self.n_labels = len(self.label_map)
        for k, v in self.label_map.items():
            self.inv_label_map[v] = k
        self.is_prepared = True

    def encode(self, data: Union[tuple, np.ndarray, pd.Series], normalize=True) -> torch.Tensor:
        """
        :param normalize: can be used to temporarily return unnormalized values
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)  # specific to the Gym class - remove once deprecated!
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = data.astype(str)
        encoded = torch.Tensor(data.map(self.label_map))

        if normalize and self.normalize:
            encoded /= self.n_labels
        if len(encoded.shape) < 2:
            encoded = encoded.unsqueeze(-1)

        return encoded

    def decode(self, encoded_values: torch.Tensor, normalize=True) -> List[object]:
        """
        :param normalize: can be used to temporarily return unnormalized values
        """
        if normalize and self.normalize:
            encoded_values *= self.n_labels
        values = encoded_values.long().squeeze().tolist()  # long() as inv_label_map expects an int key
        values = [self.inv_label_map.get(v, _UNCOMMON_WORD) for v in values]
        return values
