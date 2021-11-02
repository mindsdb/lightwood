from itertools import product
from typing import Dict, Optional

from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.helpers.general import get_group_matches
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer, AdaptiveMinMaxNormalizer


def generate_target_group_normalizers(data: Dict, tss: TimeseriesSettings, norm_class: Optional = None):
    """
    Helper function called from data_source. It generates and fits all needed normalizers for a target variable
    based on its grouped entities.
    :param data
    :param tss: TimeseriesSettings object.
    :param norm_class: type of normalizer. If not specified, defaults to `CatNormalizer` for categorical and binary targets, or `MinMaxNormalizer` for numerical targets.

    :return: modified data with dictionary with normalizers for said target variable based on some grouped-by columns
    """  # noqa
    normalizers = {}
    group_combinations = []

    # categorical normalizers
    if data['original_type'] in [dtype.categorical, dtype.binary]:
        normalizers['__default'] = CatNormalizer() if not norm_class else norm_class()
        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    # numerical normalizers, here we spawn one per each group combination
    else:
        norm_class = MinMaxNormalizer if norm_class is None else norm_class

        if data['original_type'] == dtype.tsarray:
            data['data'] = data['data'].reshape(-1, 1).astype(float)

        all_group_combinations = list(product(*[set(x) for x in data['group_info'].values()]))
        for combination in all_group_combinations:
            if combination != ():
                combination = frozenset(combination)  # freeze so that we can hash with it
                _, subset = get_group_matches(data, combination)
                if subset.size > 0:
                    if norm_class == AdaptiveMinMaxNormalizer:
                        normalizers[combination] = AdaptiveMinMaxNormalizer(tss.window, combination=combination)
                    else:
                        normalizers[combination] = norm_class(combination=combination)

                    normalizers[combination].prepare(subset)
                    group_combinations.append(combination)

        # ...plus a default one, used at inference time and fitted with all training data
        if norm_class == AdaptiveMinMaxNormalizer:
            normalizers['__default'] = AdaptiveMinMaxNormalizer(tss.window)
        else:
            normalizers['__default'] = norm_class()

        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    data['target_normalizers'] = normalizers
    data['group_combinations'] = group_combinations

    return data
