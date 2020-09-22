import torch

class BaseEncoder:
    """Base class for all encoders"""

    def __init__(self, is_target=False, secondary_type=None):
        self.is_target = is_target
        self.secondary_type = secondary_type
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, column_data):
        raise NotImplementedError

    def decode(self, encoded_data):
        raise NotImplementedError

    def to(self, device, available_devices):
        return self
