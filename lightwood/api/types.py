from os import stat
from typing import Dict, List, Union
from dataclasses import dataclass
from lightwood.helpers.log import log
from dataclasses_json import dataclass_json
from dataclasses_json.core import _asdict, Json
import json


@dataclass_json
@dataclass
class Feature:
    name: str
    data_dtype: str
    dependency: List[str] = None
    encoder: str = None


@dataclass_json
@dataclass
class Output:
    name: str
    data_dtype: str
    encoder: str = None
    models: List[str] = None
    ensemble: str = None


@dataclass_json
@dataclass
class TypeInformation:
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
    nr_rows: int
    train_std_dev: float
    # Write proper to and from dict parsing for this than switch back to using the types bellow, dataclasses_json sucks!
    train_observed_classes: object  # Union[None, List[str]]
    target_class_distribution: object  # Dict[str, float]
    histograms: object  # Dict[str, Dict[str, List[object]]]


@dataclass_json
@dataclass
class DataAnalysis:
    statistical_analysis: StatisticalAnalysis
    type_information: TypeInformation


@dataclass
class TimeseriesSettings:
    is_timeseries: bool
    order_by: List[str] = None
    window: int = None
    group_by: List[str] = None
    use_previous_target: bool = False
    nr_predictions: int = None
    historical_columns: List[str] = None
    target_type: str = ''  # @TODO: is the current setter (outside of initialization) a sane option?

    @staticmethod
    def from_dict(obj: Dict):
        if len(obj) > 0:
            for mandatory_setting in ['order_by', 'window']:
                if mandatory_setting not in obj:
                    err = f'Missing mandatory timeseries setting: {mandatory_setting}'
                    log.error(err)
                    raise Exception(err)

            timeseries_settings = TimeseriesSettings(
                is_timeseries=True,
                order_by=obj['order_by'],
                window=obj['window'],
                use_previous_target=obj.get('use_previous_target', True),
                historical_columns=[],
                nr_predictions=obj.get('nr_predictions', 1)

            )
            for setting in obj:
                timeseries_settings.__setattr__(setting, obj[setting])

        else:
            timeseries_settings = TimeseriesSettings(is_timeseries=False)

        return timeseries_settings
    
    @staticmethod
    def from_json(data: str):
        return TimeseriesSettings.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        return json.dumps(self.to_dict())

@dataclass
class ProblemDefinition:
    target: str
    nfolds: int
    pct_invalid: float
    seconds_per_model: int
    seconds_per_encoder: int
    time_aim: int
    target_weights: List[float]
    positive_domain: bool
    fixed_confidence: Union[int, float, None]
    timeseries_settings: TimeseriesSettings
    anomaly_detection: bool
    anomaly_error_rate: Union[float, None]
    anomaly_cooldown: int
    ignore_features: List[str]

    @staticmethod
    def from_dict(obj: Dict) -> None:
        target = obj['target']
        nfolds = obj.get('nfolds', 10)
        pct_invalid = obj.get('pct_invalid', 1)
        seconds_per_model = obj.get('seconds_per_model', None)
        seconds_per_encoder = obj.get('seconds_per_encoder', None)
        time_aim = obj.get('time_aim', None)
        target_weights = obj.get('target_weights', None)
        positive_domain = obj.get('positive_domain', False)
        fixed_confidence = obj.get('fixed_confidence', None)
        timeseries_settings = TimeseriesSettings.from_dict(obj.get('timeseries_settings', {}))
        anomaly_detection = obj.get('anomaly_detection', True)
        anomaly_error_rate = obj.get('anomaly_error_rate', None)
        anomaly_cooldown = obj.get('anomaly_detection', 1)
        ignore_features = obj.get('ignore_features', [])

        problem_definition = ProblemDefinition(
            target=target,
            nfolds=nfolds,
            pct_invalid=pct_invalid,
            seconds_per_model=seconds_per_model,
            seconds_per_encoder=seconds_per_encoder,
            time_aim=time_aim,
            target_weights=target_weights,
            positive_domain=positive_domain,
            fixed_confidence=fixed_confidence,
            timeseries_settings=timeseries_settings,
            anomaly_detection=anomaly_detection,
            anomaly_error_rate=anomaly_error_rate,
            anomaly_cooldown=anomaly_cooldown,
            ignore_features=ignore_features
        )

        return problem_definition

    @staticmethod
    def from_json(data: str):
        return ProblemDefinition.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        return json.dumps(self.to_dict())


@dataclass
class JsonAI:
    features: Dict[str, Feature]
    output: Output
    problem_definition: ProblemDefinition
    statistical_analysis: StatisticalAnalysis
    identifiers: Dict[str, str]
    cleaner: object = None
    splitter: object = None
    analyzer: object = None
    explainer: object = None
    imports: object = None
    timeseries_transformer: object = None
    timeseries_analyzer: object = None
    accuracy_functions: List[str] = None
    phases: Dict[str, object] = None

    @staticmethod
    def from_dict(obj: Dict) -> None:
        features = {k: Feature.from_dict(v) for k,v in obj['features'].items()} 
        output = Output.from_dict(obj['output'])
        problem_definition = ProblemDefinition.from_dict(obj['problem_definition'])
        statistical_analysis = StatisticalAnalysis.from_dict(obj['statistical_analysis']) 
        identifiers = obj['identifiers']
        cleaner = obj.get('cleaner', None)
        splitter = obj.get('splitter', None)
        analyzer = obj.get('analyzer', None)
        explainer = obj.get('explainer', None)
        imports = obj.get('imports', None)
        timeseries_transformer = obj.get('timeseries_transformer', None)
        timeseries_analyzer = obj.get('timeseries_analyzer', None)
        accuracy_functions = obj.get('accuracy_functions', None)
        phases = obj.get('phases', None)

        json_ai = JsonAI(
            features=features,
            output=output,
            problem_definition=problem_definition,
            statistical_analysis=statistical_analysis,
            identifiers=identifiers,
            cleaner=cleaner,
            splitter=splitter,
            analyzer=analyzer,
            explainer=explainer,
            imports=imports,
            timeseries_transformer=timeseries_transformer,
            timeseries_analyzer=timeseries_analyzer,
            accuracy_functions=accuracy_functions,
            phases=phases
        )

        return json_ai

    @staticmethod
    def from_json(data: str):
        return JsonAI.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        return json.dumps(self.to_dict())


@dataclass_json
@dataclass
class ModelAnalysis:
    accuracies: Dict[str, float]
    train_sample_size: int
    test_sample_size: int
    column_importances: Dict[str, float]
    confusion_matrix: object = None