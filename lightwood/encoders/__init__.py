from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.datetime.datetime import DatetimeEncoder
from lightwood.encoders.image.img_2_vec import Img2VecEncoder
from lightwood.encoders.image.nn import NnAutoEncoder
from lightwood.encoders.numeric.numeric import NumericEncoder
from lightwood.encoders.text.infersent import InferSentEncoder
from lightwood.encoders.text.distilbert import DistilBertEncoder
from lightwood.encoders.text.flair import FlairEmbeddingEncoder
from lightwood.encoders.text.short import ShortTextEncoder
from lightwood.encoders.text.vocab import VocabularyEncoder
from lightwood.encoders.text.rnn import RnnEncoder
from lightwood.encoders.categorical.onehot import OneHotEncoder
from lightwood.encoders.categorical.autoencoder import CategoricalAutoEncoder
from lightwood.encoders.time_series.ts_fresh_ts import TsFreshTsEncoder
from lightwood.encoders.time_series.rnn import RnnEncoder as TsRnnEncoder
# from lightwood.encoders.audio.amplitude_ts import AmplitudeTsEncoder
from lightwood.encoders.categorical.multihot import MultiHotEncoder

try:
    from lightwood.encoders.time_series.cesium_ts import CesiumTsEncoder
    export_cesium = True
except:
    export_cesium = False
    print('Failed to export cesium timeseires encoder')


class DateTime:
    DatetimeEncoder = DatetimeEncoder


class Image:
    Img2VecEncoder = Img2VecEncoder
    NnAutoEncoder = NnAutoEncoder


class Numeric:
    NumericEncoder = NumericEncoder


class Text:
    InferSentEncoder = InferSentEncoder
    DistilBertEncoder = DistilBertEncoder
    FlairEmbeddingEncoder = FlairEmbeddingEncoder
    ShortTextEncoder = ShortTextEncoder
    InferSentEncoder = InferSentEncoder
    RnnEncoder = RnnEncoder
    VocabularyEncoder = VocabularyEncoder

class Categorical:
    OneHotEncoder = OneHotEncoder
    CategoricalAutoEncoder = CategoricalAutoEncoder

class TimeSeries:
    TsFreshTsEncoder = TsFreshTsEncoder
    RnnEncoder = TsRnnEncoder
    if export_cesium:
        CesiumTsEncoder = CesiumTsEncoder


class BuiltinEncoders:
    DateTime = DateTime
    Image = Image
    Numeric = Numeric
    Text = Text
    Categorical = Categorical
    TimeSeries = TimeSeries
    # Audio = Audio


BUILTIN_ENCODERS = BuiltinEncoders
