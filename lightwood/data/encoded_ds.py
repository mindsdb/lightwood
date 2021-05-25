import importlib
import inspect
import copy
import random
import string
from typing import List
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lightwood.encoders.time_series.helpers.common import generate_target_group_normalizers
from lightwood.encoder.base import BaseEncoder


class EncodedDs(Dataset):
    def __init__(self, encoders: List[BaseEncoder], data_frame: pd.DataFrame, target: str):
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
        
        return ds

    def __len__(self):
        """
        return the length of the datasource (as in number of rows)
        :return: number of rows
        """
        return int(self.data_frame.shape[0])

    def __getitem__(self, idx):
        if self.cache_encoded:
            if self.cache[idx] is not None:
                return self.cache[idx]

        X = torch.FloatTensor()
        for col in self.data_frame:
            if col != target:
                encoded_tensor = self.encoders[col_name].encode(self.data_frame.iloc[idx][col_name])[0]
                X = torch.cat([X, encoded_tensor])

        Y = self.encoders[target].encode(self.data_frame.iloc[idx][col_name])[0]

        if self.cache_encoded:
            self.cache[idx] = (X, Y)
        
        return X, Y

    def get_column_original_data(self, column_name):
        return self.data_frame[column_name]

    def get_encoded_column_data(self, column_name):
        encoded_vals: List[torch.FloatTensor] = []
        for i in range(len(self)):
            encoded_vals.append(self.encoders[col_name].encode(self.data_frame[col_name]))
        return torch.stack(encoded_vals)
