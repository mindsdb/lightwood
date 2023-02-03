from typing import Dict, Optional

from lightwood.mixer.sktime import SkTime


class ARIMAMixer(SkTime):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            horizon: int,
            ts_analysis: Dict,
            model_path: str = 'statsforecast.StatsForecastAutoARIMA',
            auto_size: bool = True,
            sp: int = None,
            use_stl: bool = False,
            start_p: int = 2,
            d: Optional[int] = None,
            start_q: int = 2,
            max_p: int = 5,
            max_d: int = 2,
            max_q: int = 5,
            start_P: int = 1,
            D: Optional[int] = None,
            start_Q: int = 1,
            max_P: int = 2,
            max_D: int = 1,
            max_Q: int = 2,
            max_order: int = 5,
            seasonal: bool = True,
            stationary: bool = False,
            information_criterion: Optional[str] = None,
            alpha: float = 0.05,
            test: str = 'kpss',
            seasonal_test: Optional[str] = None,
            stepwise: bool = True,
            n_jobs: Optional[int] = None,
            start_params: Optional[list] = None,
            trend: Optional[str] = None,
            method: Optional[str] = None,
            maxiter: int = 50,
            offset_test_args: Optional[dict] = None,
            seasonal_test_args: Optional[dict] = None,
            suppress_warnings: bool = True,
            error_action: str = 'warn',
            trace: bool = False,
            random: bool = False,
            random_state: Optional[int] = None,
            n_fits: Optional[int] = None,
            out_of_sample_size: int = 0,
            scoring: str = 'mse',
            scoring_args: Optional[dict] = None,
            with_intercept: bool = True,
            update_pdq: bool = True,
            time_varying_regression: bool = False,
            enforce_stationarity: bool = True,
            enforce_invertibility: bool = True,
            simple_differencing: bool = False,
            measurement_error: bool = False,
            mle_regression: bool = True,
            hamilton_representation: bool = False,
            concentrate_scale: bool = False,
    ):
        """
        Wrapper for SkTime's AutoARIMA interface.

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
            'start_p': start_p,
            'd': d,
            'start_q': start_q,
            'max_p': max_p,
            'max_d': max_d,
            'max_q': max_q,
            'start_P': start_P,
            'D': D,
            'start_Q': start_Q,
            'max_P': max_P,
            'max_D': max_D,
            'max_Q': max_Q,
            'max_order': max_order,
            'seasonal': seasonal,
            'stationary': stationary,
            'information_criterion': information_criterion,
            'alpha': alpha,
            'test': test,
            'seasonal_test': seasonal_test,
            'stepwise': stepwise,
            'n_jobs': n_jobs,
            'start_params': start_params,
            'trend': trend,
            'method': method,
            'maxiter': maxiter,
            'offset_test_args': offset_test_args,
            'seasonal_test_args': seasonal_test_args,
            'suppress_warnings': suppress_warnings,
            'error_action': error_action,
            'trace': trace,
            'random': random,
            'random_state': random_state,
            'n_fits': n_fits,
            'out_of_sample_size': out_of_sample_size,
            'scoring': scoring,
            'scoring_args': scoring_args,
            'with_intercept': with_intercept,
            'update_pdq': update_pdq,
            'time_varying_regression': time_varying_regression,
            'enforce_stationarity': enforce_stationarity,
            'enforce_invertibility': enforce_invertibility,
            'simple_differencing': simple_differencing,
            'measurement_error': measurement_error,
            'mle_regression': mle_regression,
            'hamilton_representation': hamilton_representation,
            'concentrate_scale': concentrate_scale,
        }
        super().__init__(stop_after, target, dtype_dict, horizon, ts_analysis,
                         model_path, model_kwargs, auto_size, sp, hyperparam_search, use_stl)
        self.name = 'AutoARIMA'
        self.stable = False
