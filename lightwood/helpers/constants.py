"""
Reserved constants for Lightwood.
"""

_UNCOMMON_WORD = '__mdb_unknown_cat'
_UNCOMMON_TOKEN = 0

# For custom modules, we create a module loader with necessary imports below
IMPORT_EXTERNAL_DIRS = """
for import_dir in [os.path.join(os.path.expanduser('~/lightwood_modules'), lightwood_version.replace('.', '_')), os.path.join('/etc/lightwood_modules', lightwood_version.replace('.', '_'))]:
    if os.path.exists(import_dir) and os.access(import_dir, os.R_OK):
        for file_name in list(os.walk(import_dir))[0][2]:
            if file_name[-3:] != '.py':
                continue
            mod_name = file_name[:-3]
            loader = importlib.machinery.SourceFileLoader(mod_name,
                                                          os.path.join(import_dir, file_name))
            module = ModuleType(loader.name)
            loader.exec_module(module)
            sys.modules[mod_name] = module
            exec(f'import {mod_name}')
""" # noqa

IMPORTS = """
import lightwood
from lightwood import __version__ as lightwood_version
from lightwood.analysis import *
from lightwood.api import *
from lightwood.data import *
from lightwood.encoder import *
from lightwood.ensemble import *
from lightwood.helpers.device import *
from lightwood.helpers.general import *
from lightwood.helpers.ts import *
from lightwood.helpers.log import *
from lightwood.helpers.numeric import *
from lightwood.helpers.parallelism import *
from lightwood.helpers.seed import *
from lightwood.helpers.text import *
from lightwood.helpers.torch import *
from lightwood.mixer import *

from dataprep_ml.insights import statistical_analysis
from dataprep_ml.cleaners import cleaner
from dataprep_ml.splitters import splitter
from dataprep_ml.imputers import *

from mindsdb_evaluator import evaluate_accuracies
from mindsdb_evaluator.accuracy import __all__ as mdb_eval_accuracy_metrics

import pandas as pd
from typing import Dict, List, Union, Optional
import os
from types import ModuleType
import importlib.machinery
import sys
import time
"""
