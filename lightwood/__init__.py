import os
import logging
logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
from lightwood.api import __all__ as api_all_list
from lightwood.api import * # noqa
from lightwood import data
from lightwood.__about__ import __package_name__ as name, __version__


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
__all__ = ['data', 'name', '__version__', *api_all_list]
