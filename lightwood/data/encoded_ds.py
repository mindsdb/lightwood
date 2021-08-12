import inspect
from typing import List, Tuple
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lightwood.encoder.base import BaseEncoder


class EncodedDs(Dataset):
    def __init__(self, encoders: List[BaseEncoder], data_frame: pd.DataFrame, target: str) -> None:
        """
        Create a lightwood datasource from the data frame
        :param data_frame:
        :param config
        """
        self.data_frame = data_frame
        self.encoders = encoders
        self.target = target
        self.cache_encoded = True
        self.cache = [None] * len(self.data_frame)
        self.encoder_spans = {}
        self.input_length = 0

        # save encoder span, has to use same iterator as in __getitem__ for correct indeces
        for col in self.data_frame:
            if col != self.target and self.encoders.get(col, False):
                self.encoder_spans[col] = (self.input_length,
                                           self.input_length + self.encoders[col].output_size)
                self.input_length += self.encoders[col].output_size

    def __len__(self):
        """
        return the length of the datasource (as in number of rows)
        :return: number of rows
        """
        return int(self.data_frame.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_encoded:
            if self.cache[idx] is not None:
                return self.cache[idx]

        X = torch.FloatTensor()
        Y = torch.FloatTensor()
        for col in self.data_frame:
            if self.encoders.get(col, None):
                kwargs = {}
                if 'dependency_data' in inspect.signature(self.encoders[col].encode).parameters:
                    kwargs['dependency_data'] = {dep: [self.data_frame.iloc[idx][dep]]
                                                 for dep in self.encoders[col].dependencies}
                if hasattr(self.encoders[col], 'data_window'):
                    cols = [self.target] + [f'{self.target}_timestep_{i}'
                                            for i in range(1, self.encoders[col].data_window)]
                else:
                    cols = [col]

                data = self.data_frame[cols].iloc[idx].tolist()
                encoded_tensor = self.encoders[col].encode(data, **kwargs)[0]
                if col != self.target:
                    X = torch.cat([X, encoded_tensor])
                else:
                    Y = encoded_tensor

        if self.cache_encoded:
            self.cache[idx] = (X, Y)

        return X, Y

    def get_column_original_data(self, column_name: str) -> pd.Series:
        return self.data_frame[column_name]

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        kwargs = {}
        if 'dependency_data' in inspect.signature(self.encoders[column_name].encode).parameters:
            deps = [dep for dep in self.encoders[column_name].dependencies if dep in self.data_frame.columns]
            kwargs['dependency_data'] = {dep: self.data_frame[dep].tolist() for dep in deps}
        encoded_data = self.encoders[column_name].encode(self.data_frame[column_name], **kwargs)

        if not isinstance(encoded_data, torch.Tensor):
            raise Exception(
                f'The encoder: {self.encoders[column_name]} for column: {column_name} does not return a Tensor !')
        return encoded_data

    def get_encoded_data(self, include_target=True) -> torch.Tensor:
        encoded_dfs = []
        for col in self.data_frame.columns:
            if (include_target or col != self.target) and self.encoders.get(col, False):
                encoded_dfs.append(self.get_encoded_column_data(col))

        return torch.cat(encoded_dfs, 1)

    def clear_cache(self):
        self.cache = [None] * len(self.data_frame)


# Abstract over multiple encoded datasources as if they were a single entitiy
class ConcatedEncodedDs(EncodedDs):
    def __init__(self, encoded_ds_arr: List[EncodedDs]) -> None:
        self.encoded_ds_arr = encoded_ds_arr
        self.encoded_ds_lenghts = [len(x) for x in self.encoded_ds_arr]
        self.encoders = self.encoded_ds_arr[0].encoders
        self.encoder_spans = self.encoded_ds_arr[0].encoder_spans
        self.target = self.encoded_ds_arr[0].target

    def __len__(self):
        return max(0, np.sum(self.encoded_ds_lenghts) - 2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for ds_idx, length in enumerate(self.encoded_ds_lenghts):
            if idx - length < 0:
                return self.encoded_ds_arr[ds_idx][idx]
            else:
                idx -= length
        raise StopIteration()

    @property
    def data_frame(self):
        return pd.concat([x.data_frame for x in self.encoded_ds_arr])

    def get_column_original_data(self, column_name: str) -> pd.Series:
        encoded_df_arr = [x.get_column_original_data(column_name) for x in self.encoded_ds_arr]
        return pd.concat(encoded_df_arr)

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        encoded_df_arr = [x.get_encoded_column_data(column_name) for x in self.encoded_ds_arr]
        return torch.cat(encoded_df_arr, 0)

    def clear_cache(self):
        for ds in self.encoded_ds_arr:
            ds.clear_cache()
