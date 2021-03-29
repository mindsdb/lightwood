from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.encoders.datetime.datetime import DatetimeEncoder
from lightwood.encoders.image.img_2_vec import Img2VecEncoder
from lightwood.encoders.image.nn import NnAutoEncoder
from lightwood.encoders.numeric.numeric import NumericEncoder
from lightwood.encoders.numeric.ts_numeric import TsNumericEncoder
from lightwood.encoders.text.short import ShortTextEncoder
from lightwood.encoders.text.vocab import VocabularyEncoder
from lightwood.encoders.text.rnn import RnnEncoder as TextRnnEncoder
from lightwood.encoders.categorical.onehot import OneHotEncoder
from lightwood.encoders.categorical.autoencoder import CategoricalAutoEncoder
from lightwood.encoders.time_series.rnn import TimeSeriesEncoder as TsRnnEncoder
from lightwood.encoders.time_series.plain import TimeSeriesPlainEncoder
from lightwood.encoders.categorical.multihot import MultiHotEncoder
from lightwood.encoders.text.pretrained import PretrainedLang


class DateTime:
    DatetimeEncoder = DatetimeEncoder


class Image:
    Img2VecEncoder = Img2VecEncoder
    NnAutoEncoder = NnAutoEncoder


class Numeric:
    NumericEncoder = NumericEncoder
    TsNumericEncoder = TsNumericEncoder


class Text:
    ShortTextEncoder = ShortTextEncoder
    TextRnnEncoder = TextRnnEncoder
    VocabularyEncoder = VocabularyEncoder
    PretrainedLang = PretrainedLang

class Categorical:
    OneHotEncoder = OneHotEncoder
    CategoricalAutoEncoder = CategoricalAutoEncoder

class TimeSeries:
    TsRnnEncoder = TsRnnEncoder
    TimeSeriesPlainEncoder = TimeSeriesPlainEncoder

class BuiltinEncoders:
    DateTime = DateTime
    Image = Image
    Numeric = Numeric
    Text = Text
    Categorical = Categorical
    TimeSeries = TimeSeries
    # Audio = Audio


BUILTIN_ENCODERS = BuiltinEncoders
