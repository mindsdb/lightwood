from typing import Dict
import pandas as pd

from lightwood.api import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.encoder.time_series.helpers.common import MinMaxNormalizer, CatNormalizer, get_group_matches, generate_target_group_normalizers


def timeseries_analyzer(data: pd.DataFrame, dtype_dict: Dict[str, str], timeseries_settings: TimeseriesSettings, target: str) -> (Dict, Dict):
    info = {
        'original_type': dtype_dict[target],
        'data': data[target],
        'group_info': {gcol: data[gcol].tolist() for gcol in timeseries_settings.group_by}  # group col values
    }

    # @TODO: maybe normalizers should fit using only the training folds??
    new_data = generate_target_group_normalizers(info) if timeseries_settings.group_by else {}

    return {'target_normalizers': new_data['target_normalizers'],
            'group_combinations': new_data['group_combinations']}