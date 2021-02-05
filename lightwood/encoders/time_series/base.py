from lightwood.encoders.encoder_base import BaseEncoder


class TimeSeriesBaseEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._normalizer = None
        self._target_normalizers = []

    def prepare(self, priming_data, previous_target_data=None, feedback_hoop_function=None):
        pass