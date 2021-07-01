import enum
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
        for col in self.data_frame:
            if col != self.target:
                kwargs = {}
                if 'dependency_data' in inspect.signature(self.encoders[col].encode).parameters:
                    kwargs['dependency_data'] = {dep: [self.data_frame.iloc[idx][dep]]
                                                 for dep in self.encoders[col].dependencies}
                encoded_tensor = self.encoders[col].encode([self.data_frame.iloc[idx][col]], **kwargs)[0]
                X = torch.cat([X, encoded_tensor])
        
        Y = self.encoders[self.target].encode([self.data_frame.iloc[idx][col]])[0]

        if self.cache_encoded:
            self.cache[idx] = (X, Y)
        
        return X, Y

    def get_column_original_data(self, column_name: str) -> pd.Series:
        return self.data_frame[column_name]

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        return self.encoders[column_name].encode(self.data_frame[column_name])


# Abstract over multiple encoded datasources as if they were a single entitiy
class ConcatedEncodedDs(EncodedDs):
    def __init__(self, encoded_ds_arr: List[EncodedDs]) -> None:
        self.encoded_ds_arr = encoded_ds_arr
        self.encoded_ds_lenghts = [len(x) for x in self.encoded_ds_arr]
        self.encoders = self.encoded_ds_arr[0].encoders
        self.target = self.encoded_ds_arr[0].target

    def __len__(self):
        return np.sum(self.encoded_ds_lenghts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for ds_idx, length in enumerate(self.encoded_ds_lenghts):
            if idx - length < 0:
                return self.encoded_ds_arr[ds_idx][idx]
            else:
                idx -= length
    
    @property
    def data_frame(self):
        return pd.concat([x.data_frame for x in self.encoded_ds_arr])

    def get_column_original_data(self, column_name: str) -> pd.Series:
        encoded_df_arr = [x.get_column_original_data(column_name) for x in self.encoded_ds_arr]
        return pd.concat(encoded_df_arr)

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        encoded_df_arr = [x.get_encoded_column_data(column_name) for x in self.encoded_ds_arr]
        return torch.cat(encoded_df_arr, 0)
