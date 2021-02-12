import torch
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


class MinMaxNormalizer:
    def __init__(self, combination=(), keys=(), factor=1):
        self.scaler = MinMaxScaler()
        self.single_scaler = MinMaxScaler()  # for non-windowed arrays (when using numerical encoder)
        self.factor = factor
        self.keys = list(keys)  # columns involved in grouped-by subset dataset to normalize
        self.combination = combination  # tuple with values in those columns
        self.abs_mean = None

    def prepare(self, x):
        X = np.array([j for i in x for j in i]).reshape(-1, 1) if isinstance(x, list) else x
        X[X == None] = 0
        self.abs_mean = np.mean(np.abs(X))
        self.scaler.fit(X)
        self.single_scaler.fit(X[:, -1:])  # fit using non-windowed column data

    def encode(self, y):
        if not isinstance(y, np.ndarray) and not isinstance(y[0], list):
            y = y.reshape(-1, 1)
        return torch.Tensor(self.scaler.transform(y))

    def decode(self, y):
        return self.scaler.inverse_transform(y)

    def single_encode(self, y):
        """Variant designed for encoding a single scalar"""
        return self.single_scaler.transform(y)

    def single_decode(self, y):
        """Variant designed for decoding a single scalar"""
        return self.single_scaler.inverse_transform(y)[0][0]


class CatNormalizer:
    def __init__(self, encoder_class='one_hot'):
        if encoder_class == 'one_hot':
            self.scaler = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            self.scaler = OrdinalEncoder()
        self.unk = "<UNK>"

    def prepare(self, x):
        X = []
        for i in x:
            for j in i:
                X.append(j if j is not None else self.unk)
        self.scaler.fit(np.array(X).reshape(-1, 1))

    def encode(self, Y):
        y = np.array([[j if j is not None else self.unk for j in i] for i in Y])
        out = []
        for i in y:
            transformed = self.scaler.transform(i.reshape(-1, 1))
            if isinstance(self.scaler, OrdinalEncoder):
                transformed = transformed.flatten()
            out.append(transformed)

        return torch.Tensor(out)

    def decode(self, y):
        return [[i[0] for i in self.scaler.inverse_transform(o)] for o in y]


def get_group_matches(data, combination, keys):
    """Given a grouped-by combination, return rows of the data that match belong to it. Params:
    data: dict with data to filter and group-by columns info.
    combination: tuple with values to filter by
    keys: which column does each combination value belong to

    return: indexes for rows to normalize, data to normalize
    """
    if not combination:
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
    if data['original_type'] == 'categorical':
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data['data'])

    # numerical normalizers, here we spawn one per each group combination
    else:
        all_group_combinations = list(product(*[set(x) for x in data['group_info'].values()]))
        for combination in all_group_combinations:
            if combination != ():
                combination = frozenset(combination)  # freeze so that we can hash with it
                _, subset = get_group_matches(data, combination, data['group_info'].keys())
                if subset.size > 0:
                    normalizers[combination] = MinMaxNormalizer(combination=combination, keys=data['group_info'].keys())
                    normalizers[combination].prepare(subset)
                    group_combinations.append(combination)

        # ...plus a default one, used at inference time and fitted with all training data
        normalizers['__default'] = MinMaxNormalizer()
        normalizers['__default'].prepare(data['data'])
        group_combinations.append('__default')

    data['normalizers'] = normalizers
    data['group_combinations'] = group_combinations

    return data
