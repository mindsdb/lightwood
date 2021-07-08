import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from lightwood.api import (
    dtype,
    JsonAI,
    Output,
    Feature,
    TypeInformation,
    StatisticalAnalysis,
    generate_predictor,
    generate_json_ai,
    encode,
    analyze_dataset
)
import lightwood.data as data
from lightwood.data import infer_types, statistical_analysis


__all__ = ['data', 'infer_types', 'statistical_analysis', 'dtype', 'JsonAI', 'Output', 'Feature', 'TypeInformation', 'StatisticalAnalysis', 'generate_predictor', 'generate_json_ai', 'encode', 'analyze_dataset']
from lightwood.__about__ import __package_name__ as name, __version__
