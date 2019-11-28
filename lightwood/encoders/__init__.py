from lightwood.encoders.datetime.datetime import DatetimeEncoder
from lightwood.encoders.image.img_2_vec import Img2VecEncoder
from lightwood.encoders.image.nn import NnAutoEncoder
from lightwood.encoders.numeric.numeric import NumericEncoder
from lightwood.encoders.text.infersent import InferSentEncoder
from lightwood.encoders.text.rnn import RnnEncoder
from lightwood.encoders.time_series.cesium_ts import CesiumTsEncoder
from lightwood.encoders.categorical.onehot import OneHotEncoder
from lightwood.encoders.categorical.autoencoder import CategoricalAutoEncoder

class DateTime:
    DatetimeEncoder = DatetimeEncoder

class Image:
    Img2VecEncoder = Img2VecEncoder
    NnAutoEncoder = NnAutoEncoder

class Numeric:
    NumericEncoder = NumericEncoder

class Text:
    InferSentEncoder = InferSentEncoder
    RnnEncoder = RnnEncoder

class TimeSeries:
    CesiumTsEncoder = CesiumTsEncoder

class Categorical:
    OneHotEncoder = OneHotEncoder
    CategoricalAutoEncoder = CategoricalAutoEncoder

class BuiltinEncoders:
    DateTime = DateTime
    Image = Image
    Numeric = Numeric
    Text = Text
    TimeSeries = TimeSeries
    Categorical = Categorical

BUILTIN_ENCODERS = BuiltinEncoders
