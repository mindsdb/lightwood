from typing import List
import torch


class BaseEncoder:
    """Base class for all encoders"""
    is_target: bool
    prepared: bool

    def __init__(self, is_target=False) -> None:
        self.is_target = is_target
        self._prepared = False
        self.uses_folds = False
        self.is_nn_encoder = False
        self.dependencies = []
        self.output_size = None

    # Not all encoders need to be prepared
    def prepare(self, priming_data) -> None:
        self._prepared = True

    def encode(self, column_data) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, encoded_data) -> List[object]:
        raise NotImplementedError

    # Should work for all troch-based encoders, but custom behavior may have to be implemented for very weird models
    def to(self, device, available_devices):
        # Find all nn.Module type objects and convert them
        # @TODO: Make this work recursively
        for v in vars(self):
            attr = getattr(self, v)
            if isinstance(attr, torch.nn.Module):
                attr.to(device)
        return self
