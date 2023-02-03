from typing import Dict, Optional

from lightwood.mixer.sktime import SkTime


class ETSMixer(SkTime):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            horizon: int,
            ts_analysis: Dict,
            model_path: str = 'ets.AutoETS',
            auto_size: bool = True,
            sp: int = None,
            use_stl: bool = True,
            error: str = 'add',
            trend: Optional[str] = None,
            damped_trend: bool = False,
            seasonal: Optional[str] = None,
            initialization_method: str = 'estimated',
            initial_level: Optional[float] = None,
            initial_trend: Optional[float] = None,
            initial_seasonal: Optional[list] = None,
            bounds: Optional[dict] = None,
            start_params: Optional[list] = None,
            maxiter: int = 1000,
            auto: bool = False,
            information_criterion: str = 'aic',
            allow_multiplicative_trend: bool = False,
            restrict: bool = True,
            additive_only: bool = False,
            ignore_inf_ic: bool = True,
            n_jobs: Optional[int] = None,
            random_state: Optional[int] = None
    ):
        """
        Wrapper for SkTime's AutoETS interface.

        :param stop_after: time budget in seconds
        :param target: column containing target time series
        :param dtype_dict: data types for each dataset column
        :param horizon: forecast length
        :param ts_analysis: lightwood-produced stats about input time series
        :param auto_size: whether to filter out old data points if training split is bigger than a certain threshold (defined by the dataset sampling frequency). Enabled by default to avoid long training times in big datasets.
        :param use_stl: Whether to use de-trenders and de-seasonalizers fitted in the timeseries analysis phase.

        For the rest of the parameters, please refer to SkTime's documentation.
        """  # noqa

        hyperparam_search = False
        model_kwargs = {
            'error': error,
            'trend': trend,
            'damped_trend': damped_trend,
            'seasonal': seasonal,
            'initialization_method': initialization_method,
            'initial_level': initial_level,
            'initial_trend': initial_trend,
            'initial_seasonal': initial_seasonal,
            'bounds': bounds,
            'start_params': start_params,
            'maxiter': maxiter,
            'auto': auto,
            'information_criterion': information_criterion,
            'allow_multiplicative_trend': allow_multiplicative_trend,
            'restrict': restrict,
            'additive_only': additive_only,
            'ignore_inf_ic': ignore_inf_ic,
            'n_jobs': n_jobs,
            'random_state': random_state
        }
        super().__init__(stop_after, target, dtype_dict, horizon, ts_analysis,
                         model_path, model_kwargs, auto_size, sp, hyperparam_search, use_stl)
        self.name = 'AutoETS'
        self.stable = False
