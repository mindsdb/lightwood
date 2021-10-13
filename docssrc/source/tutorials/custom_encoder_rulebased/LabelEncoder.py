"""
2021.10.13

Create a LabelEncoder that transforms categorical data into a label.
"""
import numpy as np
from lightwood.encoder import BaseEncoder


class LabelEncoder:
    """
    Create a label representation for categorical data. The data will rely on sorted to organize the order of the labels.

    Class Attributes:
    - is_target: Whether this is used to encode the target
    - _prepared: Whether the encoder rules have been set (after ``prepare`` is called)
    - uses_subsets: Whether subsetted data is used 
    - dependencies: Any external dependencies (None)
        
    """ # noqa
    is_target: bool
    prepared: bool

    is_timeseries_encoder: bool = False
    is_trainable_encoder: bool = False

    def __init__(self, is_target: bool =False) -> None:
        """
        Initialize the Label Encoder

        :param is_target: 
        """
        self.is_target = is_target
        self._prepared = False
        self.uses_subsets = False
        self.dependencies = []
        self.output_size = None

    # Not all encoders need to be prepared
    def prepare(self, priming_data) -> None:
        self._prepared = True

    def encode(self, column_data) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, encoded_data) -> List[object]:
        raise NotImplementedError

    # Should work for all torch-based encoders, but custom behavior may have to be implemented for weird models
    def to(self, device, available_devices):
        # Find all nn.Module type objects and convert them
        # @TODO: Make this work recursively
        for v in vars(self):
            attr = getattr(self, v)
            if isinstance(attr, torch.nn.Module):
                attr.to(device)
        return self
