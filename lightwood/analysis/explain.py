from typing import Optional, List, Dict
import torch
import pandas as pd

from dataprep_ml import StatisticalAnalysis

from lightwood.helpers.log import log, timed
from lightwood.api.types import ProblemDefinition, PredictionArguments
from lightwood.helpers.ts import get_inferred_timestamps
from lightwood.analysis.base import BaseAnalysisBlock


@timed
def explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            target_name: str,
            target_dtype: str,

            problem_definition: ProblemDefinition,
            stat_analysis: StatisticalAnalysis,
            pred_args: PredictionArguments,
            runtime_analysis: Dict,

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
    predictions = predictions.reset_index(drop=True)
    data = data.reset_index(drop=True)
    tss = problem_definition.timeseries_settings
    row_insights = pd.DataFrame()
    global_insights = {}

    def _reformat_ts_columns(tss, out_df, in_df):
        if tss.is_timeseries:
            if tss.group_by:
                for col in tss.group_by:
                    out_df[f'group_{col}'] = in_df[col]

            out_df[f'order_{tss.order_by}'] = in_df[tss.order_by]
            out_df[f'order_{tss.order_by}'] = get_inferred_timestamps(
                out_df, tss.order_by, ts_analysis['deltas'], tss, stat_analysis,
                time_format=pred_args.time_format
            )
        return out_df

    if not explainer_blocks:
        predictions.rename(columns={'__mdb_original_index': 'original_index'}, inplace=True)
        predictions = _reformat_ts_columns(tss, predictions, data)
        return predictions, global_insights

    row_insights['original_index'] = data['__mdb_original_index']
    row_insights['prediction'] = predictions['prediction']

    if pred_args.predict_proba:
        for col in predictions.columns:
            if '__mdb_proba' in col:
                row_insights[col] = predictions[col]

    row_insights = _reformat_ts_columns(tss, row_insights, data)

    kwargs = {
        'data': data,
        'encoded_data': encoded_data,
        'predictions': predictions,
        'analysis': runtime_analysis,
        'target_name': target_name,
        'target_dtype': target_dtype,
        'tss': tss,
        'positive_domain': stat_analysis.positive_domain,
        'anomaly_detection': problem_definition.anomaly_detection,
        'pred_args': pred_args
    }

    # ------------------------- #
    # Call explanation blocks
    # ------------------------- #
    for block in explainer_blocks:
        log.info("The block %s is now running its explain() method", block.__class__.__name__)
        row_insights, global_insights = block.explain(row_insights, global_insights, **kwargs)

    return row_insights, global_insights
