from typing import Optional, List, Dict
import torch
import pandas as pd

import lightwood.api.json_ai
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_inferred_timestamps
from lightwood.analysis.nc.calibrate import ICP


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

    # ------------------------- #
    # Core Explanations
    # ------------------------- #

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

    # confidence estimation using calibrated inductive conformal predictors (ICPs)
    if analysis['icp']['__mdb_active']:
        calibrator = ICP()
        row_insights, global_insights = calibrator.explain(insights, **kwargs)

    # ------------------------- #
    # Additional Explanations
    # ------------------------- #
    if len(explainer_blocks) > 0:
        exec(lightwood.api.json_ai.IMPORTS_FOR_EXTERNAL_DIRS, globals())
        exec(lightwood.api.json_ai.IMPORT_EXTERNAL_DIRS, globals())

        for dirpath in explainer_blocks:
            module, block_name = dirpath.split(".")
            block = getattr(eval(module), block_name)()
            row_insights, global_insights = block.explain(insights, **kwargs)

    return row_insights
