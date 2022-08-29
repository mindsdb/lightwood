from typing import Dict, Union

from lightwood.mixer.sktime import SkTime


class ProphetMixer(SkTime):
    def __init__(self,
                 stop_after: float,
                 target: str,
                 dtype_dict: Dict[str, str],
                 horizon: int,
                 ts_analysis: Dict,
                 model_path: str = 'fbprophet.Prophet',
                 auto_size: bool = True,
                 sp: int = None,
                 hyperparam_search: bool = False,
                 use_decomposers: Dict[str, Union[int, str]] = {}
                 ):
        super().__init__(stop_after, target, dtype_dict, horizon, ts_analysis,
                         model_path, auto_size, sp, hyperparam_search, use_decomposers)
        self.stable = False
        self.model_path = model_path
        self.possible_models = [self.model_path]
        self.n_trials = len(self.possible_models)
