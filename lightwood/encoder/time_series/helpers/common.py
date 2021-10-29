from itertools import product

import numpy as np
import pandas as pd

from lightwood.api.dtype import dtype


def get_group_matches(data, combination):
    """Given a grouped-by combination, return rows of the data that match belong to it. Params:
    data: dict with data to filter and group-by columns info.
    combination: tuple with values to filter by
    return: indexes for rows to normalize, data to normalize
    """
    keys = data['group_info'].keys()  # which column does each combination value belong to

    if isinstance(data['data'], pd.Series):
        data['data'] = np.vstack(data['data'])
    if isinstance(data['data'], np.ndarray) and len(data['data'].shape) < 2:
        data['data'] = np.expand_dims(data['data'], axis=1)

    if combination == '__default':
        idxs = range(len(data['data']))
        return [idxs, np.array(data['data'])[idxs, :]]  # return all data
    else:
        all_sets = []
        for val, key in zip(combination, keys):
            all_sets.append(set([i for i, elt in enumerate(data['group_info'][key]) if elt == val]))
        if all_sets:
            idxs = list(set.intersection(*all_sets))
            return idxs, np.array(data['data'])[idxs, :]

        else:
            return [], np.array([])


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
    if data['original_type'] in [dtype.categorical, dtype.binary]:
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    # numerical normalizers, here we spawn one per each group combination
    else:
        if data['original_type'] == dtype.tsarray:
            data['data'] = data['data'].reshape(-1, 1).astype(float)

        all_group_combinations = list(product(*[set(x) for x in data['group_info'].values()]))
        for combination in all_group_combinations:
            if combination != ():
                combination = frozenset(combination)  # freeze so that we can hash with it
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
