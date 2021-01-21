import torch

class BaseEncoder:
    """Base class for all encoders"""

    def __init__(self, is_target=False):
        self.is_target = is_target
        self.original_type = None
        self.secondary_type = None
        self._prepared = False

    def prepare(self, priming_data):
        pass

    def encode(self, column_data):
        raise NotImplementedError

    def decode(self, encoded_data):
        raise NotImplementedError

    def to(self, device, available_devices):
        return self
