# Encoders which should always work
from lightwood.encoder.base import BaseEncoder
from lightwood.encoder.datetime.datetime import DatetimeEncoder
from lightwood.encoder.datetime.datetime_sin_normalizer import DatetimeNormalizerEncoder
from lightwood.encoder.image.img_2_vec import Img2VecEncoder
from lightwood.encoder.numeric.numeric import NumericEncoder
from lightwood.encoder.numeric.ts_numeric import TsNumericEncoder
from lightwood.encoder.numeric.ts_array_numeric import TsArrayNumericEncoder
from lightwood.encoder.text.short import ShortTextEncoder
from lightwood.encoder.text.vocab import VocabularyEncoder
from lightwood.encoder.text.rnn import RnnEncoder as TextRnnEncoder
from lightwood.encoder.categorical.onehot import OneHotEncoder
from lightwood.encoder.categorical.binary import BinaryEncoder
from lightwood.encoder.categorical.autoencoder import CategoricalAutoEncoder
from lightwood.encoder.time_series.rnn import TimeSeriesEncoder
from lightwood.encoder.array.array import ArrayEncoder
from lightwood.encoder.categorical.multihot import MultiHotEncoder
from lightwood.encoder.text.pretrained import PretrainedLangEncoder
from lightwood.encoder.type_encoder_maps import (Array, Binary, Categorical, Date, Datetime, Float, Image, Integer,
                                                 TimeSeries, Quantity, Rich_Text, Short_Text, Tags)


# Encoders that depend on optional dependencies
try:
    from lightwood.encoder.audio.amplitude_ts import AmplitudeTsEncoder
except Exception:
    AmplitudeTsEncoder = None


__all__ = ['BaseEncoder', 'DatetimeEncoder', 'Img2VecEncoder', 'NumericEncoder', 'TsNumericEncoder',
           'TsArrayNumericEncoder', 'ShortTextEncoder', 'VocabularyEncoder', 'TextRnnEncoder', 'OneHotEncoder',
           'CategoricalAutoEncoder', 'TimeSeriesEncoder', 'ArrayEncoder', 'MultiHotEncoder',
           'PretrainedLangEncoder', 'AmplitudeTsEncoder', 'BinaryEncoder', 'DatetimeNormalizerEncoder',
           'Array', 'TimeSeries', 'Binary', 'Categorical', 'Date', 'Datetime', 'Float', 'Image', 'Integer',
           'Quantity', 'Rich_Text', 'Short_Text', 'Tags']
