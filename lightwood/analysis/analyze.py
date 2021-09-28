from typing import Dict, List, Optional

import lightwood.api.json_ai
from lightwood.api import dtype
from lightwood.ensemble import BaseEnsemble
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.encoder.text.pretrained import PretrainedLangEncoder
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings

from lightwood.analysis.nc.calibrate import ICP
from lightwood.analysis.helpers.acc_stats import AccStats
from lightwood.analysis.helpers.feature_importance import GlobalFeatureImportance


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
    """Analyses model on a validation subset to evaluate accuracy and confidence of future predictions"""

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    is_numerical = data_type in (dtype.integer, dtype.float, dtype.array, dtype.tsarray)
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
        'accuracy_functions': accuracy_functions
    }

    # confidence estimation with inductive conformal predictors (ICPs)
    calibrator = ICP()
    runtime_analyzer = calibrator.analyze(runtime_analyzer, **kwargs)

    # validation accuracy metrics and stats (e.g. confusion matrix, histograms)
    acc_stats = AccStats()
    runtime_analyzer = acc_stats.analyze(runtime_analyzer, **kwargs)

    # global feature importance
    if not disable_column_importance:
        block = GlobalFeatureImportance()
        runtime_analyzer = block.analyze(runtime_analyzer, **kwargs)
    else:
        runtime_analyzer['column_importances'] = None

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

    # ------------------------- #
    # Additional Analysis Blocks
    # ------------------------- #
    if len(analysis_blocks) > 0:
        exec(lightwood.api.json_ai.IMPORTS_FOR_EXTERNAL_DIRS, globals())
        exec(lightwood.api.json_ai.IMPORT_EXTERNAL_DIRS, globals())

        for dirpath in analysis_blocks:
            module, block_name = dirpath.split(".")
            block = getattr(eval(module), block_name)()
            runtime_analyzer = block.analyze(runtime_analyzer, **{})

    return model_analysis, runtime_analyzer
