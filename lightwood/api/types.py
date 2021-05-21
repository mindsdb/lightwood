from typing import Dict, List
from lightwood.api import dtype
from lightwood.encoder import BaseEncoder
from lightwood.ensemble import BaseEnsemble
from lightwood.model import BaseModel

class Feature:
    name: str = None
    dtype: dtype = None
    encoder: BaseEncoder = None
    dependency: List[str] = None

class Output:
    name: str = None
    dtype: dtype = None
    encoder: BaseEncoder = None
    models: List[BaseModel] = None
    ensemble: BaseEnsemble = None

class LightwoodConfig:
    features: Dict[str, Feature] = None
    output: Output = None
    cleaner: object = None
    splitter: object = None
    analyzer: object = None

class TypeInformation:
    dtypes: Dict[str,str] = None
    additional_info: Dict[str, object] = None
    identifiers: Dict[str,object] = None

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()

class StatisticalAnalysis:
    # Addition of stuff here pending discussion with Jorge
    pass
