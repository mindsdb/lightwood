import torchvision
import numpy as np
import torch
import sys


if sys.version_info < (3, 6):
    sys.exit('Sorry, For Lightwood Python < 3.6 is not supported')

from lightwood.__about__ import __package_name__ as name, __version__
from lightwood.encoders import BUILTIN_ENCODERS
from lightwood.mixers import BUILTIN_MIXERS
from lightwood.api.predictor import Predictor
import lightwood.model_building
import lightwood.constants.lightwood as CONST
from lightwood.helpers.device import get_devices


COLUMN_DATA_TYPES = CONST.COLUMN_DATA_TYPES

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if 'cuda' in str(get_devices()[0]):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False