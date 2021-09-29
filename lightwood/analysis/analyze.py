from typing import Dict, List, Optional

from lightwood.api import dtype
from lightwood.ensemble import BaseEnsemble
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.encoder.text.pretrained import PretrainedLangEncoder
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings


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
    analysis_blocks: Optional[List[BaseAnalysisBlock]] = []
):
    """Analyses model on a validation subset to evaluate accuracy and confidence of future predictions"""

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    # encoded data representations
    encoded_train_data = ConcatedEncodedDs(train_data)
    encoded_val_data = ConcatedEncodedDs(data)
    data = encoded_val_data.data_frame
    input_cols = list([col for col in data.columns if col != target])

    # predictive task
    is_numerical = data_type in (dtype.integer, dtype.float, dtype.array, dtype.tsarray)
    is_classification = data_type in (dtype.categorical, dtype.binary)
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1
    has_pretrained_text_enc = any([isinstance(enc, PretrainedLangEncoder)
                                   for enc in encoded_train_data.encoders.values()])

    # predictions for validation dataset
    normal_predictions = predictor(encoded_val_data) if not is_classification else predictor(encoded_val_data,
                                                                                             predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    # ------------------------- #
    # Core Analysis
    # ------------------------- #
    kwargs = {
        'predictor': predictor,
        'target': target,
        'input_cols': input_cols,
        'dtype_dict': dtype_dict,
        'normal_predictions': normal_predictions,
        'data': data,
        'train_data': train_data,
        'encoded_val_data': encoded_val_data,
        'is_classification': is_classification,
        'is_numerical': is_numerical,
        'is_multi_ts': is_multi_ts,
        'stats_info': stats_info,
        'ts_cfg': ts_cfg,
        'fixed_significance': fixed_significance,
        'positive_domain': positive_domain,
        'confidence_normalizer': confidence_normalizer,
        'accuracy_functions': accuracy_functions,
        'disable_column_importance': disable_column_importance or ts_cfg.is_timeseries or has_pretrained_text_enc
    }

    # ------------------------- #
    # Run analysis blocks, both core and user-defined
    # ------------------------- #
    for block in analysis_blocks:
        runtime_analyzer = block.analyze(runtime_analyzer, **kwargs)

    # ------------------------- #
    # Populate ModelAnalysis object
    # ------------------------- #
    model_analysis = ModelAnalysis(
        accuracies=runtime_analyzer['score_dict'],
        accuracy_histogram=runtime_analyzer['acc_histogram'],
        accuracy_samples=runtime_analyzer['acc_samples'],
        train_sample_size=len(encoded_train_data),
        test_sample_size=len(encoded_val_data),
        confusion_matrix=runtime_analyzer['cm'],
        column_importances=runtime_analyzer['column_importances'],
        histograms=stats_info.histograms,
        dtypes=dtype_dict
    )

    return model_analysis, runtime_analyzer
