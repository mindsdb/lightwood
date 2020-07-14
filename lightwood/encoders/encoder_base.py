
class EncoderBase:
    """Base class for all encoders"""

    def __init__(self, is_target=False):
        self.is_target = is_target

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, column_data):
        raise NotImplementedError

    def decoder(self, encoded_data):
        raise NotImplementedError

    def to(self, device, available_devices):
        return self
