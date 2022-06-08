from itertools import product
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from lightwood.api.dtype import dtype
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
from lightwood.helpers.general import get_group_matches


def generate_target_group_normalizers(
        data: pd.DataFrame,
        target: str,
        dtype_dict: dict,
        tss: TimeseriesSettings
):
    """
    Helper function called from data_source. It generates and fits all needed normalizers for a target variable based on its grouped entities.
    
    :return: modified data with dictionary with normalizers for said target variable based on some grouped-by columns
    """  # noqa
    normalizers = {}
    group_combinations = []
    target_dtype = dtype_dict[target]
    group_values = {gcol: data[gcol].unique() for gcol in tss.group_by} if tss.group_by else {}

    # categorical normalizers
    if target_dtype in (dtype.categorical, dtype.binary, dtype.cat_tsarray):
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data)
        group_combinations.append('__default')

    # numerical normalizers, here we spawn one per each group combination
    else:
        all_group_combinations = list(product(*[set(x) for x in group_values.values()]))
        for combination in all_group_combinations:
            if combination != ():
                combination = tuple(combination)
                idxs, subset = get_group_matches(data, combination, tss.group_by)
                if subset.size > 0:
                    target_data = subset[target].values
                    if target_dtype == dtype.num_tsarray:
                        target_data = target_data.reshape(-1, 1).astype(float)

                    normalizers[combination] = MinMaxNormalizer(combination=combination)
                    normalizers[combination].prepare(target_data)
                    group_combinations.append(combination)

        # ...plus a default one, used at inference time and fitted with all training data
        normalizers['__default'] = MinMaxNormalizer()
        normalizers['__default'].prepare(data[target].values)
        group_combinations.append('__default')

    return normalizers, group_combinations
