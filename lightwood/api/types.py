from typing import Dict, List
from dataclasses import dataclass
from lightwood.helpers.log import log
from dataclasses_json import dataclass_json
from dataclasses_json.core import _asdict, Json


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
    identifiers: Dict[str, object]

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()


@dataclass_json
@dataclass
class StatisticalAnalysis:
    nr_rows: int


@dataclass
class TimeseriesSettings:
    is_timeseries: bool
    order_by: List[str] = None
    window: int = None
    group_by: List[str] = None
    use_previous_target: bool = False
    nr_predictions: int = None
    historical_columns: List[str] = None

    @staticmethod
    def from_dict(obj: Dict):
        if len(obj) > 0:
            for mandatory_setting in ['order_by', 'window']:
                err = f'Missing mandatory timeseries setting: {mandatory_setting}'
                log.error(err)
                raise Exception(err)

            timeseries_settings = TimeseriesSettings(
                is_timeseries=True,
                historical_columns=[],
                order_by=obj['order_by'],
                window=obj['window']

            )
            for setting in obj:
                timeseries_settings.__setattr__(setting, obj['setting'])

        else:
            timeseries_settings = TimeseriesSettings(is_timeseries=False)

        return timeseries_settings

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)


@dataclass
class ProblemDefinition:
    target: str
    seconds_per_model: int
    timeseries_settings: TimeseriesSettings
    pct_invalid: float

    @staticmethod
    def from_dict(obj: Dict) -> None:
        target = obj['target']
        seconds_per_model = obj.get('seconds_per_model', None)
        timeseries_settings = TimeseriesSettings.from_dict(obj.get('timeseries_settings', {}))
        pct_invalid = obj.get('pct_invalid', 1)

        problem_definition = ProblemDefinition(
            target=target,
            seconds_per_model=seconds_per_model,
            timeseries_settings=timeseries_settings,
            pct_invalid=pct_invalid,
        )

        return problem_definition

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        return _asdict(self, encode_json=encode_json)


@dataclass_json
@dataclass
class LightwoodConfig:
    features: Dict[str, Feature]
    output: Output
    problem_definition: ProblemDefinition
    cleaner: str = None
    splitter: str = None
    analyzer: str = None
    imports: str = None
