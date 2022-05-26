from typing import Dict, List, Tuple, Optional

from lightwood.helpers.log import log
from lightwood.api import dtype
from lightwood.ensemble import BaseEnsemble
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.data.encoded_ds import EncodedDs
from lightwood.encoder.text.pretrained import PretrainedLangEncoder
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings, PredictionArguments


def model_analyzer(
    predictor: BaseEnsemble,
    data: EncodedDs,
    train_data: EncodedDs,
    stats_info: StatisticalAnalysis,
    target: str,
    tss: TimeseriesSettings,
    dtype_dict: Dict[str, str],
    accuracy_functions,
    ts_analysis: Dict,
    analysis_blocks: Optional[List[BaseAnalysisBlock]] = []
) -> Tuple[ModelAnalysis, Dict[str, object]]:
    """
    Analyses model on a validation subset to evaluate accuracy, estimate feature importance and generate a
    calibration model to estimating confidence in future predictions.

    Additionally, any user-specified analysis blocks (see class `BaseAnalysisBlock`) are also called here.

    :return:
    runtime_analyzer: This dictionary object gets populated in a sequential fashion with data generated from
    any `.analyze()` block call. This dictionary object is stored in the predictor itself, and used when
    calling the `.explain()` method of all analysis blocks when generating predictions.

    model_analysis: `ModelAnalysis` object that contains core analysis metrics, not necessarily needed when predicting.
    """

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    # retrieve encoded data representations
    encoded_train_data = train_data
    encoded_val_data = data
    data = encoded_val_data.data_frame
    input_cols = list([col for col in data.columns if col != target])

    # predictive task
    is_numerical = data_type in (dtype.integer, dtype.float, dtype.num_tsarray, dtype.quantity)
    is_classification = data_type in (dtype.categorical, dtype.binary, dtype.cat_tsarray)
    is_multi_ts = tss.is_timeseries and tss.horizon > 1
    has_pretrained_text_enc = any([isinstance(enc, PretrainedLangEncoder)
                                   for enc in encoded_train_data.encoders.values()])

    # raw predictions for validation dataset
    args = {} if not is_classification else {"predict_proba": True}
    normal_predictions = predictor(encoded_val_data, args=PredictionArguments.from_dict(args))
    normal_predictions = normal_predictions.set_index(data.index)

    # ------------------------- #
    # Run analysis blocks, both core and user-defined
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
        'tss': tss,
        'ts_analysis': ts_analysis,
        'accuracy_functions': accuracy_functions,
        'has_pretrained_text_enc': has_pretrained_text_enc
    }

    for block in analysis_blocks:
        log.info("The block %s is now running its analyze() method", block.__class__.__name__)
        runtime_analyzer = block.analyze(runtime_analyzer, **kwargs)

    # ------------------------- #
    # Populate ModelAnalysis object
    # ------------------------- #
    model_analysis = ModelAnalysis(
        accuracies=runtime_analyzer.get('score_dict', {}),
        accuracy_histogram=runtime_analyzer.get('acc_histogram', {}),
        accuracy_samples=runtime_analyzer.get('acc_samples', {}),
        train_sample_size=len(encoded_train_data),
        test_sample_size=len(encoded_val_data),
        confusion_matrix=runtime_analyzer.get('cm', []),
        column_importances=runtime_analyzer.get('column_importances', {}),
        histograms=stats_info.histograms,
        dtypes=dtype_dict,
        submodel_data=predictor.submodel_data
    )

    return model_analysis, runtime_analyzer
