# TODO: type hint the returns

from typing import Dict, List, Optional, Union
import sys

from lightwood.api.dtype import dtype

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from dataclasses import dataclass
from lightwood.helpers.log import log
from dataclasses_json import dataclass_json
from dataclasses_json.core import _asdict, Json
import json


# See: https://www.python.org/dev/peps/pep-0589/ for how this works
# Not very intuitive but very powerful abstraction, might be useful in other places (@TODO)
class Module(TypedDict):
    """
    Modules are the blocks of code that end up being called from the JSON AI, representing either object instantiations or function calls.

    :param module: Name of the module (function or class name)
    :param args: Argument to pass to the function or constructor
    """ # noqa
    module: str
    args: Dict[str, str]


@dataclass_json
@dataclass
class TypeInformation:
    """
    For a dataset, provides information on columns types, how they're used, and any other potential identifiers.

    TypeInformation is generated within ``data.infer_types``, where small samples of each column are evaluated in a custom framework to understand what kind of data type the model is. The user may override data types, but it is recommended to do so within a JSON-AI config file.

    :param dtypes: For each column's name, the associated data type inferred.
    :param additional_info: Any possible sub-categories or additional descriptive information.
    :param identifiers: Columns within the dataset highly suspected of being identifiers or IDs. These do not contain informatic value, therefore will be ignored in subsequent training/analysis procedures unless manually indicated.
    """ # noqa

    dtypes: Dict[str, str]
    additional_info: Dict[str, object]
    identifiers: Dict[str, str]

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()


@dataclass_json
@dataclass
class StatisticalAnalysis:
    """
    The Statistical Analysis data class allows users to consider key descriptors of their data using simple \
        techniques such as histograms, mean and standard deviation, word count, missing values, and any detected bias\
             in the information.

    :param nr_rows: Number of rows (samples) in the dataset
    :param df_target_stddev: The standard deviation of the target of the dataset
    :param train_observed_classes:
    :param target_class_distribution:
    :param target_weights: What weight the analysis suggests to assign each class by in the case of classification problems. Note: target_weights in the problem definition overides this
    :param histograms:
    :param buckets:
    :param missing:
    :param distinct:
    :param bias:
    :param avg_words_per_sentence:
    :param positive_domain:
    :param ts_stats:
    """ # noqa

    nr_rows: int
    df_target_stddev: Optional[float]
    train_observed_classes: object  # Union[None, List[str]]
    target_class_distribution: object  # Dict[str, float]
    target_weights: object  # Dict[str, float]
    histograms: object  # Dict[str, Dict[str, List[object]]]
    buckets: object  # Dict[str, Dict[str, List[object]]]
    missing: object
    distinct: object
    bias: object
    avg_words_per_sentence: object
    positive_domain: bool
    ts_stats: dict


@dataclass_json
@dataclass
class DataAnalysis:
    """
    Data Analysis wraps :class: `.StatisticalAnalysis` and :class: `.TypeInformation` together. Further details can be seen in their respective documentation references.
    """ # noqa

    statistical_analysis: StatisticalAnalysis
    type_information: TypeInformation


@dataclass
class TimeseriesSettings:
    """
    For time-series specific problems, more specific treatment of the data is necessary. The following attributes \
        enable time-series tasks to be carried out properly.

    :param is_timeseries: Whether the input data should be treated as time series; if true, this flag is checked in \
        subsequent internal steps to ensure processing is appropriate for time-series data.
    :param order_by: Column by which the data should be ordered.
    :param group_by: Optional list of columns by which the data should be grouped. Each different combination of values\
         for these columns will yield a different series.
    :param window: The temporal horizon (number of rows) that a model intakes to "look back" into when making a\
         prediction, after the rows are ordered by the order_by column and split into groups if applicable.
    :param horizon: The number of points in the future that predictions should be made for, defaults to 1. Once \
        trained, the model will be able to predict up to this many points into the future.
    :param historical_columns: The temporal dynamics of these columns will be used as additional context to train the \
        time series predictor. Note that a non-historical column shall still be used to forecast, but without \
            considering their change through time.
    :param target_type: Automatically inferred dtype of the target (e.g. `dtype.integer`, `dtype.float`).
    :param use_previous_target: Use the previous values of the target column to generate predictions. Defaults to True.
    :param allow_incomplete_history: whether predictions can be made for rows with incomplete historical context (i.e. less than `window` rows have been observed for the datetime that has to be forecasted).
    :param eval_cold_start: whether to include predictions with incomplete history (thus part of the cold start region for certain mixers) when evaluating mixer scores with the validation dataset.
    :param interval_periods: tuple of tuples with user-provided period lengths for time intervals. Default values will be added for intervals left unspecified. For interval options, check the `timeseries_analyzer.detect_period()` method documentation. e.g.: (('daily', 7),).
    """  # noqa

    is_timeseries: bool
    order_by: str = None
    window: int = None
    group_by: List[str] = None
    use_previous_target: bool = True
    horizon: int = None
    historical_columns: List[str] = None
    target_type: str = (
        ""  # @TODO: is the current setter (outside of initialization) a sane option?
        # @TODO: George: No, I don't think it is, we need to pass this some other way
    )
    allow_incomplete_history: bool = True
    eval_cold_start: bool = True
    interval_periods: tuple = tuple()

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a TimeseriesSettings object from python dictionary specifications.

        :param: obj: A python dictionary with the necessary representation for time-series. The only mandatory columns are ``order_by`` and ``window``.

        :returns: A populated ``TimeseriesSettings`` object.
        """ # noqa
        if len(obj) > 0:
            for mandatory_setting, etype in zip(["order_by", "window"], [str, int]):
                if mandatory_setting not in obj:
                    err = f"Missing mandatory timeseries setting: {mandatory_setting}"
                    log.error(err)
                    raise Exception(err)
                if obj[mandatory_setting] and not isinstance(obj[mandatory_setting], etype):
                    err = f"Wrong type for mandatory timeseries setting '{mandatory_setting}': found '{type(obj[mandatory_setting])}', expected '{etype}'"  # noqa
                    log.error(err)
                    raise Exception(err)

            timeseries_settings = TimeseriesSettings(
                is_timeseries=True,
                order_by=obj["order_by"],
                window=obj["window"],
                use_previous_target=obj.get("use_previous_target", True),
                historical_columns=[],
                horizon=obj.get("horizon", 1),
                allow_incomplete_history=obj.get('allow_incomplete_history', True),
                eval_cold_start=obj.get('eval_cold_start', True),
                interval_periods=obj.get('interval_periods', tuple(tuple()))
            )
            for setting in obj:
                timeseries_settings.__setattr__(setting, obj[setting])

        else:
            timeseries_settings = TimeseriesSettings(is_timeseries=False)

        return timeseries_settings

    @staticmethod
    def from_json(data: str):
        """
        Creates a TimeseriesSettings object from JSON specifications via python dictionary.

        :param: data: JSON-config file with necessary Time-series specifications

        :returns: A populated ``TimeseriesSettings`` object.
        """
        return TimeseriesSettings.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a dictionary from ``TimeseriesSettings`` object

        :returns: A python dictionary containing the ``TimeSeriesSettings`` specifications.
        """
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        """
        Creates JSON config from TimeseriesSettings object
        :returns: The JSON config syntax containing the ``TimeSeriesSettings`` specifications.
        """
        return json.dumps(self.to_dict())


@dataclass
class ProblemDefinition:
    """
    The ``ProblemDefinition`` object indicates details on how the models that predict the target are prepared. \
        The only required specification from a user is the ``target``, which indicates the column within the input \
        data that the user is trying to predict. Within the ``ProblemDefinition``, the user can specify aspects \
        about how long the feature-engineering preparation may take, and nuances about training the models.

    :param target: The name of the target column; this is the column that will be used as the goal of the prediction.
    :param pct_invalid: Number of data points maximally tolerated as invalid/missing/unknown. \
        If the data cleaning process exceeds this number, no subsequent steps will be taken.
    :param unbias_target: all classes are automatically weighted inverse to how often they occur
    :param seconds_per_mixer: Number of seconds maximum to spend PER mixer trained in the list of possible mixers.
    :param seconds_per_encoder: Number of seconds maximum to spend when training an encoder that requires data to \
    learn a representation.
    :param expected_additional_time: Time budget for non-encoder/mixer tasks \
    (ex: data analysis, pre-processing, model ensembling or model analysis)
    :param time_aim: Time budget (in seconds) to train all needed components for the predictive tasks, including \
        encoders and models.
    :param target_weights: indicates to the accuracy functions how much to weight every target class.
    :param positive_domain: For numerical taks, force predictor output to be positive (integer or float).
    :param timeseries_settings: TimeseriesSettings object for time-series tasks, refer to its documentation for \
         available settings.
    :param anomaly_detection: Whether to conduct unsupervised anomaly detection; currently supported only for time-\
        series.
    :param ignore_features: The names of the columns the user wishes to ignore in the ML pipeline. Any column name \
        found in this list will be automatically removed from subsequent steps in the ML pipeline.
    :param use_default_analysis: whether default analysis blocks are enabled.
    :param fit_on_all: Whether to fit the model on the held-out validation data. Validation data is strictly \
        used to evaluate how well a model is doing and is NEVER trained. However, in cases where users anticipate new \
            incoming data over time, the user may train the model further using the entire dataset.
    :param strict_mode: crash if an `unstable` block (mixer, encoder, etc.) fails to run.
    :param seed_nr: custom seed to use when generating a predictor from this problem definition.
    """

    target: str
    pct_invalid: float
    unbias_target: bool
    seconds_per_mixer: Optional[int]
    seconds_per_encoder: Optional[int]
    expected_additional_time: Optional[int]
    time_aim: Optional[float]
    target_weights: Optional[List[float]]
    positive_domain: bool
    timeseries_settings: TimeseriesSettings
    anomaly_detection: bool
    use_default_analysis: bool
    ignore_features: List[str]
    fit_on_all: bool
    strict_mode: bool
    seed_nr: int

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a ProblemDefinition object from a python dictionary with necessary specifications.

        :param obj: A python dictionary with the necessary features for the ``ProblemDefinition`` class.
        Only requires ``target`` to be specified.

        :returns: A populated ``ProblemDefinition`` object.
        """
        target = obj['target']
        pct_invalid = obj.get('pct_invalid', 2)
        unbias_target = obj.get('unbias_target', True)
        seconds_per_mixer = obj.get('seconds_per_mixer', None)
        seconds_per_encoder = obj.get('seconds_per_encoder', None)
        expected_additional_time = obj.get('expected_additional_time', None)

        time_aim = obj.get('time_aim', None)
        if time_aim is not None and time_aim < 10:
            log.warning(f'Your specified time aim of {time_aim} is too short. Setting it to 10 seconds.')

        target_weights = obj.get('target_weights', None)
        positive_domain = obj.get('positive_domain', False)
        timeseries_settings = TimeseriesSettings.from_dict(obj.get('timeseries_settings', {}))
        anomaly_detection = obj.get('anomaly_detection', False)
        ignore_features = obj.get('ignore_features', [])
        fit_on_all = obj.get('fit_on_all', True)
        use_default_analysis = obj.get('use_default_analysis', True)
        strict_mode = obj.get('strict_mode', True)
        seed_nr = obj.get('seed_nr', 1)
        problem_definition = ProblemDefinition(
            target=target,
            pct_invalid=pct_invalid,
            unbias_target=unbias_target,
            seconds_per_mixer=seconds_per_mixer,
            seconds_per_encoder=seconds_per_encoder,
            expected_additional_time=expected_additional_time,
            time_aim=time_aim,
            target_weights=target_weights,
            positive_domain=positive_domain,
            timeseries_settings=timeseries_settings,
            anomaly_detection=anomaly_detection,
            ignore_features=ignore_features,
            use_default_analysis=use_default_analysis,
            fit_on_all=fit_on_all,
            strict_mode=strict_mode,
            seed_nr=seed_nr
        )

        return problem_definition

    @staticmethod
    def from_json(data: str):
        """
        Creates a ProblemDefinition Object from JSON config file.

        :param data:

        :returns: A populated ProblemDefinition object.
        """
        return ProblemDefinition.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary from the ProblemDefinition object

        :returns: A python dictionary
        """
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        """
        Creates a JSON config from the ProblemDefinition object

        :returns: TODO
        """
        return json.dumps(self.to_dict())


@dataclass
class JsonAI:
    """
    The JsonAI Class allows users to construct flexible JSON config to specify their ML pipeline. JSON-AI follows a \
    recipe of how to pre-process data, construct features, and train on the target column. To do so, the following \
    specifications are required internally.

    :param encoders: A dictionary of the form: `column_name -> encoder module`
    :param dtype_dict: A dictionary of the form: `column_name -> data type`
    :param dependency_dict: A dictionary of the form: `column_name -> list of columns it depends on`
    :param model: The ensemble and its submodels
    :param problem_definition: The ``ProblemDefinition`` criteria.
    :param identifiers: A dictionary of column names and respective data types that are likely identifiers/IDs within the data. Through the default cleaning process, these are ignored.
    :param cleaner: The Cleaner object represents the pre-processing step on a dataframe. The user can specify custom subroutines, if they choose, on how to handle preprocessing. Alternatively, "None" suggests Lightwood's default approach in ``data.cleaner``.
    :param splitter: The Splitter object is the method in which the input data is split into training/validation/testing data.
    :param analyzer: The Analyzer object is used to evaluate how well a model performed on the predictive task.
    :param explainer: The Explainer object deploys explainability tools of interest on a model to indicate how well a model generalizes its predictions.
    :param imputers: A list of objects that will impute missing data on each column. They are called inside the cleaner.
    :param analysis_blocks: The blocks that get used in both analysis and inference inside the analyzer and explainer blocks.
    :param timeseries_transformer: Procedure used to transform any timeseries task dataframe into the format that lightwood expects for the rest of the pipeline.  
    :param timeseries_analyzer: Procedure that extracts key insights from any timeseries in the data (e.g. measurement frequency, target distribution, etc).
    :param accuracy_functions: A list of performance metrics used to evaluate the best mixers.
    """ # noqa

    encoders: Dict[str, Module]
    dtype_dict: Dict[str, dtype]
    dependency_dict: Dict[str, List[str]]
    model: Dict[str, Module]
    problem_definition: ProblemDefinition
    identifiers: Dict[str, str]
    cleaner: Optional[Module] = None
    splitter: Optional[Module] = None
    analyzer: Optional[Module] = None
    explainer: Optional[Module] = None
    imputers: Optional[List[Module]] = None
    analysis_blocks: Optional[List[Module]] = None
    timeseries_transformer: Optional[Module] = None
    timeseries_analyzer: Optional[Module] = None
    accuracy_functions: Optional[List[str]] = None

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a JSON-AI object from dictionary specifications of the JSON-config.
        """
        encoders = obj["encoders"]
        dtype_dict = obj["dtype_dict"]
        dependency_dict = obj["dependency_dict"]
        model = obj["model"]
        problem_definition = ProblemDefinition.from_dict(obj["problem_definition"])
        identifiers = obj["identifiers"]
        cleaner = obj.get("cleaner", None)
        splitter = obj.get("splitter", None)
        analyzer = obj.get("analyzer", None)
        explainer = obj.get("explainer", None)
        imputers = obj.get("imputers", None)
        analysis_blocks = obj.get("analysis_blocks", None)
        timeseries_transformer = obj.get("timeseries_transformer", None)
        timeseries_analyzer = obj.get("timeseries_analyzer", None)
        accuracy_functions = obj.get("accuracy_functions", None)

        json_ai = JsonAI(
            encoders=encoders,
            dtype_dict=dtype_dict,
            dependency_dict=dependency_dict,
            model=model,
            problem_definition=problem_definition,
            identifiers=identifiers,
            cleaner=cleaner,
            splitter=splitter,
            analyzer=analyzer,
            explainer=explainer,
            imputers=imputers,
            analysis_blocks=analysis_blocks,
            timeseries_transformer=timeseries_transformer,
            timeseries_analyzer=timeseries_analyzer,
            accuracy_functions=accuracy_functions,
        )

        return json_ai

    @staticmethod
    def from_json(data: str):
        """ Creates a JSON-AI object from JSON config"""
        return JsonAI.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary with necessary modules within the ML pipeline specified from the JSON-AI object.

        :returns: A python dictionary that has the necessary components of the ML pipeline for a given dataset.
        """
        as_dict = _asdict(self, encode_json=encode_json)
        for k in list(as_dict.keys()):
            if k == "features":
                feature_dict = {}
                for name in self.features:
                    feature_dict[name] = self.features[name].to_dict()
                as_dict[k] = feature_dict
            if as_dict[k] is None:
                del as_dict[k]
        return as_dict

    def to_json(self) -> Dict[str, Json]:
        """
        Creates JSON config to represent the necessary modules within the ML pipeline specified from the JSON-AI object.

        :returns: A JSON config that has the necessary components of the ML pipeline for a given dataset.
        """
        return json.dumps(self.to_dict(), indent=4)


@dataclass_json
@dataclass
class SubmodelData:
    name: str
    accuracy: float
    is_best: bool


@dataclass_json
@dataclass
class ModelAnalysis:
    """
    The ``ModelAnalysis`` class stores useful information to describe a model and understand its predictive performance on a validation dataset.
    For each trained ML algorithm, we store:

    :param accuracies: Dictionary with obtained values for each accuracy function (specified in JsonAI)
    :param accuracy_histogram: Dictionary with histograms of reported accuracy by target value.
    :param accuracy_samples: Dictionary with sampled pairs of observed target values and respective predictions.
    :param train_sample_size: Size of the training set (data that parameters are updated on)
    :param test_sample_size: Size of the testing set (explicitly held out)
    :param column_importances: Dictionary with the importance of each column for the model, as estimated by an approach that closely follows a leave-one-covariate-out strategy.
    :param confusion_matrix: A confusion matrix for the validation dataset.
    :param histograms: Histogram for each dataset feature.
    :param dtypes: Inferred data types for each dataset feature.

    """ # noqa

    accuracies: Dict[str, float]
    accuracy_histogram: Dict[str, list]
    accuracy_samples: Dict[str, list]
    train_sample_size: int
    test_sample_size: int
    column_importances: Dict[str, float]
    confusion_matrix: object
    histograms: object
    dtypes: object
    submodel_data: List[SubmodelData]


@dataclass
class PredictionArguments:
    """
    This class contains all possible arguments that can be passed to a Lightwood predictor at inference time.
    On each predict call, all arguments included in a parameter dictionary will update the respective fields
    in the `PredictionArguments` instance that the predictor will have.
    
    :param predict_proba: triggers (where supported) predictions in raw probability output form. I.e. for classifiers,
    instead of returning only the predicted class, the output additionally includes the assigned probability for
    each class.   
    :param all_mixers: forces an ensemble to return predictions emitted by all its internal mixers. 
    :param fixed_confidence: Used in the ICP analyzer module, specifies an `alpha` fixed confidence so that predictions, in average, are correct `alpha` percent of the time. For unsupervised anomaly detection, this also translates into the expected error rate. Bounded between 0.01 and 0.99 (respectively implies wider and tighter bounds, all other parameters being equal).
    :param anomaly_cooldown: Sets the minimum amount of timesteps between consecutive firings of the the anomaly \
        detector.
    """  # noqa

    predict_proba: bool = True
    all_mixers: bool = False
    fixed_confidence: Union[int, float, None] = None
    anomaly_cooldown: int = 1
    forecast_offset: int = 0

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a ``PredictionArguments`` object from a python dictionary with necessary specifications.

        :param obj: A python dictionary with the necessary features for the ``PredictionArguments`` class.

        :returns: A populated ``PredictionArguments`` object.
        """

        # maybe this should be stateful instead, and save the latest used value for each field?
        predict_proba = obj.get('predict_proba', PredictionArguments.predict_proba)
        all_mixers = obj.get('all_mixers', PredictionArguments.all_mixers)
        fixed_confidence = obj.get('fixed_confidence', PredictionArguments.fixed_confidence)
        anomaly_cooldown = obj.get('anomaly_cooldown', PredictionArguments.anomaly_cooldown)
        forecast_offset = obj.get('forecast_offset', PredictionArguments.forecast_offset)

        pred_args = PredictionArguments(
            predict_proba=predict_proba,
            all_mixers=all_mixers,
            fixed_confidence=fixed_confidence,
            anomaly_cooldown=anomaly_cooldown,
            forecast_offset=forecast_offset,
        )

        return pred_args

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary from the ``PredictionArguments`` object

        :returns: A python dictionary
        """
        return _asdict(self, encode_json=encode_json)
