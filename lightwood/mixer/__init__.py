from lightwood.mixer.base import BaseMixer
from lightwood.mixer.unit import Unit
from lightwood.mixer.neural import Neural
from lightwood.mixer.neural_ts import NeuralTs
from lightwood.mixer.lightgbm import LightGBM
from lightwood.mixer.random_forest import RandomForest
from lightwood.mixer.lightgbm_array import LightGBMArray
from lightwood.mixer.sktime import SkTime
from lightwood.mixer.arima import ARIMAMixer
from lightwood.mixer.ets import ETSMixer
from lightwood.mixer.gluonts import GluonTSMixer
from lightwood.mixer.regression import Regression

try:
    from lightwood.mixer.qclassic import QClassic
except Exception:
    QClassic = None

try:
    from lightwood.mixer.nhits import NHitsMixer
except Exception:
    NHitsMixer = None

try:
    from lightwood.mixer.prophet import ProphetMixer
except Exception:
    ProphetMixer = None

__all__ = ['BaseMixer', 'Neural', 'NeuralTs', 'LightGBM', 'RandomForest', 'LightGBMArray', 'Unit', 'Regression',
           'SkTime', 'QClassic', 'ProphetMixer', 'ETSMixer', 'ARIMAMixer', 'NHitsMixer', 'GluonTSMixer']
