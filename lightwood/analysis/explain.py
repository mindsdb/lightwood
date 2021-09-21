from typing import Optional, List, Dict
import torch
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_inferred_timestamps
from lightwood.analysis.nc.calibrate import icp_explain


def explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            timeseries_settings: TimeseriesSettings,
            analysis: Dict,
            target_name: str,
            target_dtype: str,

            positive_domain: bool,  # @TODO: pass these bools to the block constructor so that they are not needed here
            fixed_confidence: float,
            anomaly_detection: bool,

            # forces specific confidence level in ICP
            anomaly_error_rate: float,

            # ignores anomaly detection for N steps after an
            # initial anomaly triggers the cooldown period;
            # implicitly assumes series are regularly spaced
            anomaly_cooldown: int,

            explainer_blocks: Optional[List] = [],
            ts_analysis: Optional[Dict] = {}
            ):

    data = data.reset_index(drop=True)

    insights = pd.DataFrame()
    insights['prediction'] = predictions['prediction']

    if target_name in data.columns:
        insights['truth'] = data[target_name]
    else:
        insights['truth'] = [None] * len(predictions['prediction'])

    if timeseries_settings.is_timeseries:
        if timeseries_settings.group_by:
            for col in timeseries_settings.group_by:
                insights[f'group_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            insights[f'order_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            insights[f'order_{col}'] = get_inferred_timestamps(
                insights, col, ts_analysis['deltas'], timeseries_settings)

    # confidence estimation using calibrated inductive conformal predictors (ICPs)
    if analysis['icp']['__mdb_active']:
        insights = icp_explain(data,
                               encoded_data,
                               predictions,
                               analysis,
                               insights,
                               target_name,
                               target_dtype,
                               timeseries_settings,
                               positive_domain,
                               fixed_confidence,
                               anomaly_detection,
                               anomaly_error_rate,
                               anomaly_cooldown
                               )

    # user explainer blocks
    for block in explainer_blocks:
        insights = block.explain(insights, **{})

    return insights
