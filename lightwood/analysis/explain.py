from typing import Optional, List, Dict
import torch
import pandas as pd

from lightwood.helpers.log import log
from lightwood.api.types import TimeseriesSettings, PredictionArguments
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
            anomaly_detection: bool,
            pred_args: PredictionArguments,

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
    predictions = predictions.reset_index(drop=True)

    row_insights = pd.DataFrame()
    global_insights = {}
    row_insights['original_index'] = data['__mdb_original_index']
    row_insights['prediction'] = predictions['prediction']

    if pred_args.predict_proba:
        for col in predictions.columns:
            if '__mdb_proba' in col:
                row_insights[col] = predictions[col]

    if timeseries_settings.is_timeseries:
        if timeseries_settings.group_by:
            for col in timeseries_settings.group_by:
                row_insights[f'group_{col}'] = data[col]

        row_insights[f'order_{timeseries_settings.order_by}'] = data[timeseries_settings.order_by]
        row_insights[f'order_{timeseries_settings.order_by}'] = get_inferred_timestamps(
            row_insights, timeseries_settings.order_by, ts_analysis['deltas'], timeseries_settings)

    kwargs = {
        'data': data,
        'encoded_data': encoded_data,
        'predictions': predictions,
        'analysis': analysis,
        'target_name': target_name,
        'target_dtype': target_dtype,
        'tss': timeseries_settings,
        'positive_domain': positive_domain,
        'anomaly_detection': anomaly_detection,
        'pred_args': pred_args
    }

    # ------------------------- #
    # Call explanation blocks
    # ------------------------- #
    for block in explainer_blocks:
        log.info("The block %s is now running its explain() method", block.__class__.__name__)
        row_insights, global_insights = block.explain(row_insights, global_insights, **kwargs)

    return row_insights, global_insights
