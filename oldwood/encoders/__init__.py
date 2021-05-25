from lightwood.encoder.base import BaseEncoder
from lightwood.encoder.datetime.datetime import DatetimeEncoder
from lightwood.encoder.image.img_2_vec import Img2VecEncoder
from lightwood.encoder.image.nn import NnAutoEncoder
from lightwood.encoder.numeric.numeric import NumericEncoder
from lightwood.encoder.numeric.ts_numeric import TsNumericEncoder
from lightwood.encoder.text.short import ShortTextEncoder
from lightwood.encoder.text.vocab import VocabularyEncoder
from lightwood.encoder.text.rnn import RnnEncoder as TextRnnEncoder
from lightwood.encoder.categorical.onehot import OneHotEncoder
from lightwood.encoder.categorical.autoencoder import CategoricalAutoEncoder
from lightwood.encoder.time_series.rnn import TimeSeriesEncoder as TsRnnEncoder
from lightwood.encoder.time_series.plain import TimeSeriesPlainEncoder
from lightwood.encoder.categorical.multihot import MultiHotEncoder
from lightwood.encoder.text.pretrained import PretrainedLang


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
