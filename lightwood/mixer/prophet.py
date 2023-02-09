from typing import Dict, Optional, Union

import pandas as pd

from lightwood.mixer.sktime import SkTime


class ProphetMixer(SkTime):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            horizon: int,
            ts_analysis: Dict,
            auto_size: bool = True,
            use_stl: bool = False,
            add_seasonality: Optional[dict] = None,
            add_country_holidays: Optional[dict] = None,
            growth: str = 'linear',
            growth_floor: float = 0,
            growth_cap: Optional[float] = None,
            changepoints: Optional[list] = None,
            n_changepoints: int = 25,
            changepoint_range: float = 0.8,
            yearly_seasonality: Union[str, bool, int] = 'auto',
            weekly_seasonality: Union[str, bool, int] = 'auto',
            daily_seasonality: Union[str, bool, int] = 'auto',
            holidays: Optional[pd.DataFrame] = None,
            seasonality_mode: str = 'additive',
            seasonality_prior_scale: float = 10.0,
            holidays_prior_scale: float = 10.0,
            changepoint_prior_scale: float = 0.05,
            mcmc_samples: int = 0,
            alpha: float = 0.05,
            uncertainty_samples: int = 1000,
    ):
        """
        Wrapper for SkTime's Prophet interface.
         
        :param stop_after: time budget in seconds
        :param target: column containing target time series
        :param dtype_dict: data types for each dataset column
        :param horizon: forecast length
        :param ts_analysis: lightwood-produced stats about input time series
        :param auto_size: whether to filter out old data points if training split is bigger than a certain threshold (defined by the dataset sampling frequency). Enabled by default to avoid long training times in big datasets.
        :param use_stl: Whether to use de-trenders and de-seasonalizers fitted in the timeseries analysis phase.
        
        For the rest of the parameters, please refer to SkTime's documentation.
        """  # noqa

        # overrides
        sp = 1
        hyperparam_search = False
        model_path = 'fbprophet.Prophet'
        model_kwargs = {
            'add_seasonality': add_seasonality,
            'add_country_holidays': add_country_holidays,
            'growth': growth,
            'growth_floor': growth_floor,
            'growth_cap': growth_cap,
            'changepoints': changepoints,
            'n_changepoints': n_changepoints,
            'changepoint_range': changepoint_range,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'holidays': holidays,
            'seasonality_mode': seasonality_mode,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'changepoint_prior_scale': changepoint_prior_scale,
            'mcmc_samples': mcmc_samples,
            'alpha': alpha,
            'uncertainty_samples': uncertainty_samples,
        }

        # setup sktime base mixer
        super().__init__(stop_after, target, dtype_dict, horizon, ts_analysis,
                         model_path, model_kwargs, auto_size, sp, hyperparam_search, use_stl)
        self.name = 'Prophet'
        self.stable = False
