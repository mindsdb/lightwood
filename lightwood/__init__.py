import sys


if sys.version_info < (3,6):
    sys.exit('Sorry, For Lightwood Python < 3.6 is not supported')

import lightwood.constants.lightwood as  CONST
import lightwood.model_building
from lightwood.api.predictor import Predictor
from lightwood.mixers import BUILTIN_MIXERS
from lightwood.encoders import BUILTIN_ENCODERS
from lightwood.__about__ import __package_name__ as name, __version__


COLUMN_DATA_TYPES = CONST.COLUMN_DATA_TYPES
