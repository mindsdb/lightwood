import torch
import numpy as np

from lightwood.encoder.numeric import NumericEncoder
from lightwood.encoder.time_series.helpers.common import AdaptiveMinMaxNormalizer
from lightwood.helpers.log import log


class TsNumericEncoder(NumericEncoder):
    """
    Variant of vanilla numerical encoder, supports dynamic mean re-scaling
    """

    def __init__(self, is_target: bool = False, positive_domain: bool = False, grouped_by: list = [],
                 prev_target: str = None):
        super(TsNumericEncoder, self).__init__(is_target=is_target, positive_domain=positive_domain)
        # time series normalization params
        self.normalizers = None
        self.group_combinations = None
        self.grouped_by = grouped_by
        self.prev_target = f'__mdb_ts_previous_{prev_target}' if prev_target else None
        self.dependencies = [*self.grouped_by, self.prev_target] if self.grouped_by or self.prev_target else []
        self.output_size = 2 if is_target else 3

    def encode(self, data, dependency_data={}):
        """dependency_data: dict with grouped_by column info,
        to retrieve the correct normalizer for each datum"""
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if not dependency_data:
            group_data = {k: {'__default': [None] * len(data)} for k in self.grouped_by}
            historicals = [[None] * len(data)]
        else:
            group_data = {k: dependency_data[k] for k in self.grouped_by} if self.grouped_by else \
                {'__default': [None] * len(data)}
            historicals = dependency_data[self.prev_target] if self.prev_target is not None else [[None] * len(data)]

        ret = []

        for real, group, historical in zip(data, list(zip(*group_data.values())), historicals):

            try:
                real = float(real)
            except Exception:
                try:
                    real = float(real.replace(',', '.'))
                except Exception:
                    real = None
            if self.is_target:
                vector = [0] * 2
                if self.grouped_by and self.normalizers is not None:
                    try:
                        normalizer = self.normalizers[frozenset(group)]
                        if isinstance(normalizer, AdaptiveMinMaxNormalizer):
                            mean = normalizer.get_mavg(np.array([historical]).astype(float))
                        else:
                            mean = normalizer.abs_mean
                    except KeyError:
                        normalizer = self.normalizers['__default']  # novel group-by, use default normalizer
                        if isinstance(normalizer, AdaptiveMinMaxNormalizer):
                            mean = normalizer.mavg
                        else:
                            mean = normalizer.abs_mean
                else:
                    mean = self._abs_mean
                if real is not None and mean > 0:
                    vector[0] = 1 if real < 0 and not self.positive_domain else 0
                    vector[1] = real / mean
                else:
                    raise Exception(f'Can\'t encode target value: {real}')

            else:
                vector = [0] * 3
                try:
                    if real is not None:
                        vector[0] = 1
                        vector[1] = 1 if real < 0 and not self.positive_domain else 0
                        vector[2] = real / self._abs_mean
                except Exception as e:
                    log.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(vector)

        return torch.Tensor(ret)

    def decode(self, encoded_values, dependency_data=None):
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        ret = []
        if not dependency_data:
            group_data = {'__default': [None] * len(encoded_values)}
            historicals = [[None] * len(encoded_values)]
        else:
            if self.grouped_by:
                group_data = {k: [e[0] if isinstance(e, list) else e for e in dependency_data[k]]
                              for k in self.grouped_by}
            else:
                group_data = {'__default': [None] * len(encoded_values)}
            historicals = dependency_data[self.prev_target] if self.prev_target is not None else \
                [[None] * len(encoded_values)]

        if isinstance(encoded_values, torch.Tensor):
            encoded_values = encoded_values.tolist()

        for vector, group, historical in zip(encoded_values, list(zip(*group_data.values())), historicals):
            if self.is_target:
                if np.isnan(vector[0]) or vector[0] == float('inf') or np.isnan(vector[1]) or vector[1] == float('inf'):
                    log.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10, 63)
                else:
                    if group is not None and self.normalizers is not None:
                        try:
                            normalizer = self.normalizers[frozenset(group)]
                            if isinstance(normalizer, AdaptiveMinMaxNormalizer):
                                if historical:
                                    mean = normalizer.get_mavg(np.array([historical]).astype(float)).flatten()[0]
                                else:
                                    mean = normalizer.mavg.flatten()[0]
                                if mean is None:
                                    mean = normalizer.abs_mean
                            else:
                                mean = normalizer.abs_mean
                        except KeyError:
                            normalizer = self.normalizers['__default']  # novel group-by, use default normalizer
                            if isinstance(normalizer, AdaptiveMinMaxNormalizer):
                                if historical:
                                    mean = normalizer.get_mavg(np.array([historical]).astype(float)).flatten()[0]
                                else:
                                    mean = normalizer.mavg.flatten()[0]
                                if mean is None:
                                    mean = normalizer.abs_mean
                            else:
                                mean = normalizer.abs_mean
                    else:
                        mean = self._abs_mean

                    real_value = vector[1] * mean

                    if self.positive_domain:
                        real_value = abs(real_value)

                    if self._type == 'int':
                        real_value = int(round(real_value, 0))

            else:
                if vector[0] < 0.5:
                    ret.append(None)
                    continue

                real_value = vector[2] * self._abs_mean

                if self._type == 'int':
                    real_value = round(real_value)

            ret.append(real_value)
        return ret
