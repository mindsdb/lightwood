from typing import Dict, List
import json
from dataclasses import dataclass
from lightwood.helpers.log import log
from dataclasses_json import dataclass_json


@dataclass
class Feature:
    name: str
    data_dtype: str
    dependency: List[str]
    encoder: str

    def __init__(self):
        self.dependency = []


@dataclass
class Output:
    name: str
    data_dtype: str
    encoder: str
    models: List[str]
    ensemble: str

    def __init__(self):
        self.models = []


@dataclass_json
@dataclass
class LightwoodConfig:
    features: Dict[str, Feature]
    output: Output
    cleaner: str
    splitter: str
    analyzer: str
    imports: str

    def __init__(self, dict_obj: Dict[str, object] = None):
        if dict_obj is None:
            self.features = dict()
            self.imports = ''
        else:
            # TODO impl serialization for classes that contain sub-objects that need it
            self.features = dict_obj['features']
            self.output = dict_obj['output']
            self.cleaner = dict_obj['cleaner']
            self.splitter = dict_obj['splitter']
            self.analyzer = dict_obj['analyzer']

    @staticmethod
    def loads(json_str):
        config_json = json.loads(json_str)
        return LightwoodConfig(config_json)

    def dumps(self):
        return ''


@dataclass
class TypeInformation:
    dtypes: Dict[str, str]
    additional_info: Dict[str, object]
    identifiers: Dict[str, object]

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()


@dataclass
class StatisticalAnalysis:
    # Addition of stuff here pending discussion with Jorge
    pass


@dataclass
class TimeseriesSettings:
    is_timeseries: bool
    group_by: List[str]
    order_by: List[str]
    window: int
    use_previous_target: bool
    nr_predictions: int
    historical_columns: List[str]

    def __init__(self, obj: Dict) -> None:
        if len(obj) > 0:
            self.is_timeseries = True
            for mandatory_setting in ['order_by', 'window']:
                err = f'Missing mandatory timeseries setting: {mandatory_setting}'
                log.error(err)
                raise Exception(err)
            for setting in obj:
                self.__setattr__(setting, obj['setting'])
        else:
            self.is_timeseries = False


@dataclass_json
@dataclass
class ProblemDefinition:
    time_per_model: int
    timeseries_settings: TimeseriesSettings

    def __init__(self, obj: Dict) -> None:
        self.time_per_model = obj.get('time_per_model', 18446744073709551615)
        self.timeseries_settings = TimeseriesSettings(obj.get('timeseries_settings', {}))
