from copy import deepcopy
from typing import Optional, Tuple, Dict
from types import SimpleNamespace

import pandas as pd
import numpy as np

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.log import log


class MAvg(BaseAnalysisBlock):
    """
    Wrapper analysis block for the 'Robbie'...
    """

    def __init__(self, deps: Optional[Tuple] = ()):
        super().__init__(deps=deps)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        df: contains the entire time series to analyze.
        box_dim: width & height relative to the observed magnitudes in the series (height) and the amount of observations (width).

        The box is positioned at the end of the time series, and it alerts the user if the series "escapes" or "exits" the box via the y axis at any given point.

        TODO: Automated box_dim param estimation. How??
        """  # noqa
        # TODO: box_dim (User defined)
        box_dim = (0.1, 0.1)

        log.info('Anomaly detection...')
        ns = SimpleNamespace(**kwargs)
        df = deepcopy(ns.data)

        x, y = box_dim

        assert 0 <= x <= 1
        assert 0 <= y <= 1

        n_points = round(len(df) * x)
        obs_magnitude = abs(df[ns.target_name].max() - df[ns.target_name].min())

        ref_point = df[ns.target_name].iloc[-1]

        ub = ref_point + (obs_magnitude / 2) * y
        lb = ref_point - (obs_magnitude / 2) * y

        sub_df = df[ns.target_name].iloc[-n_points:].values

        flag = False

        lower_filter = np.where(lb <= sub_df, True, False)
        upper_filter = np.where(sub_df <= ub, True, False)

        if np.all([lower_filter, upper_filter]):
            flag = True

        global_insights['mavg_flag'] = flag

        return row_insights, global_insights
