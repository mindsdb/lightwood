from typing import Union, List, Dict

import torch
import numpy as np
import pandas as pd

from lightwood.encoder.numeric import NumericEncoder


class TsNumericEncoder(NumericEncoder):
    """
    Variant of vanilla numerical encoder, supports dynamic mean re-scaling
    """
    is_timeseries_encoder: bool = True

    def __init__(self, is_target: bool = False, positive_domain: bool = False, grouped_by=None):
        super(TsNumericEncoder, self).__init__(is_target=is_target, positive_domain=positive_domain)
        # time series normalization params
        self.normalizers = None
        self.group_combinations = None
        self.dependencies = grouped_by
        self.output_size = 1

    def encode(self, data: Union[np.ndarray, pd.Series], dependency_data: Dict[str, List[pd.Series]] = {}):
        """
        :param data: A pandas series containing the numbers to be encoded
        :param dependency_data: dict with grouped_by column info, to retrieve the correct normalizer for each datum

        :returns: A torch tensor with the representations of each number
        """  # noqa
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if not dependency_data:
            dependency_data = {'__default': [None] * len(data)}

        if isinstance(data, pd.Series):
            data = data.values

        # get array of series-wise observed means
        if self.normalizers is None:
            means = np.full((len(data)), fill_value=self._abs_mean)
        else:
            # use global mean as default for novel series
            means = np.full((len(data)), fill_value=self.normalizers['__default'].abs_mean)

            def _get_group_mean(group) -> float:
                if (group, ) in self.normalizers:
                    return self.normalizers[(group, )].abs_mean
                else:
                    return self.normalizers['__default'].abs_mean

            for i, group in enumerate(list(zip(*dependency_data.values()))):  # TODO: support multigroup
                if group[0] is not None:
                    means = np.vectorize(_get_group_mean, otypes=[float])(group[0].values)

        if len(data.shape) > 1 and data.shape[1] > 1:
            if len(means.shape) == 1:
                means = np.expand_dims(means, 1)
            means = np.repeat(means, data.shape[1], axis=1)

        def _norm_fn(x: float, mean: float) -> float:
            return x / mean

        # nones = np.vectorize(self._none_fn, otypes=[float])(data)  # TODO
        encoded = np.vectorize(_norm_fn, otypes=[float])(data, means)
        # encoded[nones] = 0  # if measurement is None, it is zeroed out  # TODO

        # TODO: mask for where mean is 0, then pass real as-is

        return torch.Tensor(encoded).unsqueeze(1)

    def decode(self, encoded_values: torch.Tensor, decode_log: bool = None, dependency_data=None):
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        assert isinstance(encoded_values, torch.Tensor), 'It is not a tensor!'  # TODO: debug purposes
        assert not decode_log  # TODO: debug purposes

        if not dependency_data:
            dependency_data = {'__default': [None] * len(encoded_values)}

        # force = True prevents side effects on the original encoded_values
        ev = encoded_values.numpy(force=True)

        # set global mean as default
        ret = np.full((ev.shape[0],), dtype=float, fill_value=self._abs_mean)

        # TODO: perhaps capture nan, infs, etc and set to pow(10,63)?

        # set means array
        if self.normalizers is None:
            means = np.full((ev.shape[0],), fill_value=self._abs_mean)
        else:
            means = np.full((len(encoded_values)), fill_value=self.normalizers['__default'].abs_mean)
            for i, group in enumerate(list(zip(*dependency_data.values()))):
                if group is not None:
                    if tuple(group) in self.normalizers:
                        means[i] = self.normalizers[tuple(group)].abs_mean
                    else:
                        means[i] = self.normalizers['__default'].abs_mean
                else:
                    means[i] = self._abs_mean

        # set real value
        real_value = np.multiply(ev[:].reshape(-1,), means)
        valid_mask = np.ones_like(real_value, dtype=bool)

        # final filters
        if self.positive_domain:
            real_value = abs(real_value)

        ret[valid_mask] = real_value[valid_mask]  # TODO probably not needed

        return ret.tolist()
