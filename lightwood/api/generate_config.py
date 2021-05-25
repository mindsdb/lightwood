from lightwood.api import LightwoodConfig, TypeInformation, StatisticalAnalysis1, Feature, Output
from lightwood.api import dtype
from lightwood.encoders import (
    NumericEncoder,
    CategoricalAutoEncoder,
    MultiHotEncoder,
    DatetimeEncoder,
    Img2VecEncoder,
    TsRnnEncoder,
    ShortTextEncoder,
    VocabularyEncoder,
    PretrainedLang,
    OneHotEncoder,
    BaseEncoder
)
from lightwood.model import LightGBM, Nn, BaseModel
from lightwood.ensemble import BestOf, BaseEnsemble
from lightwood.data import cleaner, splitter
from lightwood.analysis import model_analyzer


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

def lookup_encoder(col_dtype: dtype, is_target: bool):
    encoder_lookup = {
        dtype.integer: NumericEncoder,
        dtype.float: NumericEncoder,
        dtype.binary: OneHotEncoder,
        dtype.categorical: CategoricalAutoEncoder,
        dtype.tags: MultiHotEncoder,
        dtype.date: DatetimeEncoder,
        dtype.datetime: DatetimeEncoder,
        dtype.image: Img2VecEncoder,
        dtype.rich_text: PretrainedLang,
        dtype.short_text: ShortTextEncoder,
        dtype.array: TsRnnEncoder,
    }

    target_encoder_lookup_override = {
        ColumnDataTypes.rich_text: VocabularyEncoder
    }

    encoder_class = encoder_lookup[col_dtype]
    if is_target:
        if col_dtype in target_encoder_lookup_override:
            encoder_class = target_encoder_lookup_override[col_dtype]
    return encoder_class

def create_feature(name: str, col_dtype: dtype) -> Feature:
    feature = Feature()
    feature.name = name
    feature.dtype = dtype
    feature.encoder = lookup_encoder(col_dtype, False)
    return feature

def generate_config(target: str, type_information: TypeInformation, statistical_analysis: StatisticalAnalysis) -> LightwoodConfig:

    lightwood_config = LightwoodConfig()
    for col_name, col_dtype in type_information.dtypes.items():
        if type_information.identifiers[col_name] is None and col_dtype not in (dtype.invalid, invalid.empty) and col_name != target:
            lightwood_config.features[col_name] = create_feature(col_name, col_dtype)

    output = Output()
    output.name = target
    output.dtype = type_information.dtypes[target]
    output.encoder = lookup_encoder(type_information.dtypes[target], True)
    output.models = [Nn, LightGBM]
    output.ensemble = BestOf
    lightwood_config.output = output

    lightwood_config.cleaner = cleaner
    lightwood_config.splitter = splitter
    lightwood_config.analyzer = model_analyzer

    return lightwood_config
