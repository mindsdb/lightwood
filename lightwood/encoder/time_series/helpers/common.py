
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
from lightwood.helpers.ts import get_group_matches


def generate_target_group_normalizers(
        data: pd.DataFrame,
        target: str,
        dtype_dict: dict,
        groups: list,
        tss: TimeseriesSettings
):
    """
    Helper function called from data_source. It generates and fits all needed normalizers for a target variable based on its grouped entities.
    
    :return: modified data with dictionary with normalizers for said target variable based on some grouped-by columns
    """  # noqa
    normalizers = {}
    target_dtype = dtype_dict[target]

    # categorical normalizers
    if target_dtype in (dtype.categorical, dtype.binary, dtype.cat_tsarray):
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data)

    # numerical normalizers, here we spawn one per each group combination
    else:
        for combination in groups:
            if combination not in ('__default', ()):
                combination = tuple(combination)
                idxs, subset = get_group_matches(data, combination, tss.group_by)
                if subset.shape[0] > 0:
                    target_data = subset[target].values
                    if target_dtype == dtype.num_tsarray:
                        target_data = target_data.reshape(-1, 1).astype(float)

                    normalizers[combination] = MinMaxNormalizer(combination=combination)
                    normalizers[combination].prepare(target_data)

        # ...plus a default one, used at inference time and fitted with all training data
        normalizers['__default'] = MinMaxNormalizer()
        normalizers['__default'].prepare(data[target].values)

    return normalizers
