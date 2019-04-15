import sys
if sys.version_info < (3,3):
    sys.exit('Sorry, For Lightwood Python < 3.3 is not supported')

import lightwood.constants.lightwood as  CONST
from lightwood.api.predictor import Predictor
