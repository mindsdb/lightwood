from itertools import product
from lightwood.api.dtype import dtype
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer
from lightwood.helpers.general import get_group_matches


def generate_target_group_normalizers(data):
    """
    Helper function called from data_source. It generates and fits all needed normalizers for a target variable
    based on its grouped entities.
    :param data:
    :return: modified data with dictionary with normalizers for said target variable based on some grouped-by columns
    """
    normalizers = {}
    group_combinations = []

    # categorical normalizers
    if data['original_type'] in (dtype.categorical, dtype.binary, dtype.cat_tsarray):
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    # numerical normalizers, here we spawn one per each group combination
    else:
        if data['original_type'] == dtype.num_tsarray:
            data['data'] = data['data'].reshape(-1, 1).astype(float)

        all_group_combinations = list(product(*[set(x) for x in data['group_info'].values()]))
        for combination in all_group_combinations:
            if combination != ():
                combination = tuple(combination)
                _, subset = get_group_matches(data, combination)
                if subset.size > 0:
                    normalizers[combination] = MinMaxNormalizer(combination=combination)
                    normalizers[combination].prepare(subset)
                    group_combinations.append(combination)

        # ...plus a default one, used at inference time and fitted with all training data
        normalizers['__default'] = MinMaxNormalizer()
        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    data['target_normalizers'] = normalizers
    data['group_combinations'] = group_combinations

    return data
