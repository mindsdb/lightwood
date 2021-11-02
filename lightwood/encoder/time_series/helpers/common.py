from itertools import product
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings


class MinMaxNormalizer:
    def __init__(self, combination=()):
        self.scaler = MinMaxScaler()
        self.abs_mean = None
        self.combination = combination  # tuple with values in grouped-by columns
        self.output_size = 1

    def prepare(self, x: np.ndarray) -> None:
        # @TODO: streamline input type
        if isinstance(x[0], list):
            x = np.vstack(x)
        if isinstance(x[0], torch.Tensor):
            x = torch.stack(x).numpy()
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)

        x = x.astype(float)
        x[x == None] = 0 # noqa
        self.abs_mean = np.mean(np.abs(x))
        self.scaler.fit(x)

    def encode(self, y: np.ndarray) -> torch.Tensor:
        if isinstance(y[0], list):
            y = np.vstack(y)
        if isinstance(y[0], torch.Tensor):
            y = torch.stack(y).numpy()
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=1)

        shape = y.shape
        y = y.astype(float).reshape(-1, self.scaler.n_features_in_)
        out = torch.reshape(torch.Tensor(self.scaler.transform(y)), shape)
        return out

    def decode(self, y):
        return self.scaler.inverse_transform(y)


class AdaptiveMinMaxNormalizer(MinMaxNormalizer):
    def __init__(self, window_size, combination=()):
        super().__init__(combination)
        self.window_size = window_size
        self.mavg = None

    def get_mavg(self, arr):
        arr[np.isnan(arr)] = self.abs_mean  # if no info, assume absolute mean seen at training

        if arr.shape[1] > self.window_size:
            arr = arr[:, :self.window_size]
        else:
            arr = np.pad(arr, max(0, (arr.shape[1] - 1) - self.window_size), constant_values=self.abs_mean)

        mean = arr.mean(axis=1).reshape(-1, 1)

        return np.clip(mean, 1, max(1, max(mean)))

    def encode(self, y) -> torch.Tensor:
        if isinstance(y[0], list):
            y = np.vstack(y)
        if isinstance(y[0], torch.Tensor):
            y = torch.stack(y).numpy()
        if len(y.shape) < 2:
            y = np.expand_dims(y, axis=1)

        y = y.astype(float)
        self.mavg = self.get_mavg(y)
        y /= np.repeat(self.mavg, y.shape[1], axis=1)
        return torch.Tensor(y)

    def decode(self, y):
        if self.mavg is None:
            mavg = self.get_mavg(y)  # get mavg from current data
        else:
            mavg = self.mavg
        decoded = y * mavg
        return decoded


class CatNormalizer:
    def __init__(self, encoder_class='one_hot'):
        self.encoder_class = encoder_class
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
        self.output_size = len(self.scaler.categories_[0]) if self.encoder_class == 'one_hot' else 1

    def encode(self, Y):
        y = np.array([[str(j) if j is not None else self.unk for j in i] for i in Y])
        out = []
        for i in y:
            transformed = self.scaler.transform(i.reshape(-1, 1))
            if isinstance(self.scaler, OrdinalEncoder):
                transformed = transformed.flatten()
            out.append(transformed)

        return torch.Tensor(out)

    def decode(self, y):
        return [[i[0] for i in self.scaler.inverse_transform(o)] for o in y]


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
