from typing import Dict, List

from lightwood.encoder.base import BaseEncoder
from lightwood.mixer.neural import Neural
from lightwood.mixer.helpers.qclassic_net import QClassicNet
from lightwood.api.types import TimeseriesSettings


class QClassic(Neural):
    # wrapper class to be combined with Neural class when performance stabilizes
    def __init__(
            self, stop_after: float, target: str, dtype_dict: Dict[str, str],
            input_cols: List[str],
            timeseries_settings: TimeseriesSettings, target_encoder: BaseEncoder, net: str, fit_on_dev: bool,
            search_hyperparameters: bool):
        super().__init__(stop_after, target, dtype_dict,
                         input_cols, timeseries_settings, target_encoder,
                         net, fit_on_dev, search_hyperparameters)

        quantum_nets = {"QClassic": QClassicNet}
        self.net_class = quantum_nets.get(net, QClassicNet)
