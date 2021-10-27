from typing import Optional, List, Dict
import torch
import pandas as pd

from lightwood.helpers.log import log
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_inferred_timestamps
from lightwood.analysis.base import BaseAnalysisBlock


def explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            timeseries_settings: TimeseriesSettings,
            analysis: Dict,
            target_name: str,
            target_dtype: str,

            positive_domain: bool,  # @TODO: pass inside a {} with params for each block to avoid signature overload
            fixed_confidence: float,
            anomaly_detection: bool,

            # forces specific confidence level in ICP
            anomaly_error_rate: float,

            # ignores anomaly detection for N steps after an
            # initial anomaly triggers the cooldown period;
            # implicitly assumes series are regularly spaced
            anomaly_cooldown: int,

            explainer_blocks: Optional[List[BaseAnalysisBlock]] = [],
            ts_analysis: Optional[Dict] = {}
            ):
    """
    This procedure runs at the end of every normal `.predict()` call. Its goal is to generate prediction insights,
    potentially using information generated at the model analysis stage (e.g. confidence estimation).

    As in `analysis()`, any user-specified analysis blocks (see class `BaseAnalysisBlock`) are also called here.

    :return:
    row_insights: a DataFrame containing predictions and all generated insights at a row-level.
    """

    # ------------------------- #
    # Setup base insights
    # ------------------------- #
    data = data.reset_index(drop=True)

    row_insights = pd.DataFrame()
    global_insights = {}
    row_insights['prediction'] = predictions['prediction']

    if target_name in data.columns:
        row_insights['truth'] = data[target_name]
    else:
        row_insights['truth'] = [None] * len(predictions['prediction'])

    if timeseries_settings.is_timeseries:
        if timeseries_settings.group_by:
            for col in timeseries_settings.group_by:
                row_insights[f'group_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            row_insights[f'order_{col}'] = data[col]

        for col in timeseries_settings.order_by:
            row_insights[f'order_{col}'] = get_inferred_timestamps(
                row_insights, col, ts_analysis['deltas'], timeseries_settings)

    kwargs = {
        'data': data,
        'encoded_data': encoded_data,
        'predictions': predictions,
        'analysis': analysis,
        'target_name': target_name,
        'target_dtype': target_dtype,
        'tss': timeseries_settings,
        'positive_domain': positive_domain,
        'fixed_confidence': fixed_confidence,
        'anomaly_detection': anomaly_detection,
        'anomaly_error_rate': anomaly_error_rate,
        'anomaly_cooldown': anomaly_cooldown
    }

    # ------------------------- #
    # Call explanation blocks
    # ------------------------- #
    for block in explainer_blocks:
        log.info("The block %s is now running its explain() method", block.__class__.__name__)
        row_insights, global_insights = block.explain(row_insights, global_insights, **kwargs)

    return row_insights, global_insights
