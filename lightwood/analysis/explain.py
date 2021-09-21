import torch
import pandas as pd

from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.ts import get_inferred_timestamps
from lightwood.analysis.nc.calibrate import icp_explain


def explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            timeseries_settings: TimeseriesSettings,
            analysis: dict,
            target_name: str,
            target_dtype: str,
            positive_domain: bool,
            fixed_confidence: float,
            anomaly_detection: bool,

            # forces specific confidence level in ICP
            anomaly_error_rate: float,

            # ignores anomaly detection for N steps after an
            # initial anomaly triggers the cooldown period;
            # implicitly assumes series are regularly spaced
            anomaly_cooldown: int,

            ts_analysis: dict = None
            ):

    # @TODO: check not quick_predict
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

    # Make sure the target and real values are of an appropriate type
    if timeseries_settings.is_timeseries and timeseries_settings.nr_predictions > 1:
        # Array output that are not of type <array> originally are odd and I'm not sure how to handle them
        # Or if they even need handling yet
        pass
    elif target_dtype in (dtype.integer):
        insights['prediction'] = insights['prediction'].astype(int)
        insights['upper'] = insights['upper'].astype(int)
        insights['lower'] = insights['lower'].astype(int)
    elif target_dtype in (dtype.float):
        insights['prediction'] = insights['prediction'].astype(float)
        insights['upper'] = insights['upper'].astype(float)
        insights['lower'] = insights['lower'].astype(float)
    elif target_dtype in (dtype.short_text, dtype.rich_text, dtype.binary, dtype.categorical):
        insights['prediction'] = insights['prediction'].astype(str)

    return insights
