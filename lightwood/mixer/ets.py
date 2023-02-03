from typing import Dict

from lightwood.mixer.sktime import SkTime


class ETSMixer(SkTime):
    def __init__(self,
                 stop_after: float,
                 target: str,
                 dtype_dict: Dict[str, str],
                 horizon: int,
                 ts_analysis: Dict,
                 model_path: str = 'ets.AutoETS',
                 auto_size: bool = True,
                 sp: int = None,
                 use_stl: bool = True
                 ):
        hyperparam_search = False
        model_kwargs = {}
        super().__init__(stop_after, target, dtype_dict, horizon, ts_analysis,
                         model_path, model_kwargs, auto_size, sp, hyperparam_search, use_stl)
        self.name = 'AutoETS'
        self.stable = False
