from typing import Dict, List, Optional

import numpy as np

from lightwood.api import dtype
from lightwood.ensemble import BaseEnsemble
from lightwood.helpers.general import evaluate_accuracy
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.encoder.text.pretrained import PretrainedLangEncoder
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings

from lightwood.analysis.nc.calibrate import icp_calibration
from lightwood.analysis.helpers.acc_stats import AccStats
from lightwood.analysis.helpers.feature_importance import compute_global_importance


def model_analyzer(
    predictor: BaseEnsemble,
    data: List[EncodedDs],
    train_data: List[EncodedDs],
    stats_info: StatisticalAnalysis,
    target: str,
    ts_cfg: TimeseriesSettings,
    dtype_dict: Dict[str, str],
    disable_column_importance: bool,
    fixed_significance: float,
    positive_domain: bool,
    confidence_normalizer: bool,
    accuracy_functions,
    analysis_blocks: Optional = []
):
    """Analyses model on a validation fold to evaluate accuracy and confidence of future predictions"""

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
    is_classification = data_type in (dtype.categorical, dtype.binary)
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    # encoded data representations
    encoded_train_data = ConcatedEncodedDs(train_data)
    encoded_val_data = ConcatedEncodedDs(data)
    data = encoded_val_data.data_frame

    # additional flags
    has_pretrained_text_enc = any([isinstance(enc, PretrainedLangEncoder)
                                   for enc in encoded_train_data.encoders.values()])
    disable_column_importance = disable_column_importance or ts_cfg.is_timeseries or has_pretrained_text_enc

    input_cols = list([col for col in data.columns if col != target])
    normal_predictions = predictor(encoded_val_data) if not is_classification else predictor(
        encoded_val_data, predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    # core analysis methods:
    # 1. confidence estimation with inductive conformal predictors (ICPs)
    icp_output, result_df = icp_calibration(
        predictor,
        target,
        dtype_dict,
        normal_predictions,
        data,
        train_data,
        encoded_val_data,
        is_classification,
        is_numerical,
        is_multi_ts,
        stats_info,
        ts_cfg,
        fixed_significance,
        positive_domain,
        confidence_normalizer,
    )
    runtime_analyzer = {**runtime_analyzer, **icp_output}

    # 2. accuracy metric for validation data
    score_dict = evaluate_accuracy(data, normal_predictions['prediction'], target, accuracy_functions)
    normal_accuracy = np.mean(list(score_dict.values()))

    # 3. global feature importance
    if not disable_column_importance:
        column_importances = compute_global_importance(
            predictor,
            input_cols,
            target,
            data,
            encoded_val_data,
            normal_accuracy,
            accuracy_functions,
            is_classification,
            ts_cfg
        )
    else:
        column_importances = None

    # 4. validation stats (e.g. confusion matrix, histograms)
    acc_stats = AccStats(dtype_dict=dtype_dict, target=target, buckets=stats_info.buckets)
    acc_stats.fit(data, normal_predictions, conf=result_df)
    bucket_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats(
        is_classification=is_classification, is_numerical=is_numerical)
    runtime_analyzer['bucket_accuracy'] = bucket_accuracy

    model_analysis = ModelAnalysis(
        accuracies=score_dict,
        accuracy_histogram=accuracy_histogram,
        accuracy_samples=accuracy_samples,
        train_sample_size=len(encoded_train_data),
        test_sample_size=len(encoded_val_data),
        confusion_matrix=cm,
        column_importances=column_importances,
        histograms=stats_info.histograms,
        dtypes=dtype_dict
    )

    # user analysis blocks
    for block in analysis_blocks:
        runtime_analyzer = block.compute(runtime_analyzer, **{})

    return model_analysis, runtime_analyzer
