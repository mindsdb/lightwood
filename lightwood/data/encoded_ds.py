import inspect
from typing import List, Tuple, Dict
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lightwood.encoder.base import BaseEncoder


class EncodedDs(Dataset):
    def __init__(self, encoders: Dict[str, BaseEncoder], data_frame: pd.DataFrame, target: str) -> None:
        """
        Create a Lightwood datasource from a data frame and some encoders. This class inherits from `torch.utils.data.Dataset`.
        
        Note: normal behavior is to cache encoded representations to avoid duplicated computations. If you want an option to disable, this please open an issue.
         
        :param encoders: dictionary of Lightwood encoders used to encode the data per each column.
        :param data_frame: original dataframe.
        :param target: name of the target column to predict.
        """  # noqa
        self.data_frame = data_frame
        self.encoders = encoders
        self.target = target
        self.encoder_spans = {}
        self.input_length = 0  # feature tensor dim

        # save encoder span, has to use same iterator as in __getitem__ for correct indeces
        for col in self.data_frame:
            if col != self.target and self.encoders.get(col, False):
                self.encoder_spans[col] = (self.input_length,
                                           self.input_length + self.encoders[col].output_size)
                self.input_length += self.encoders[col].output_size

        # if cache enabled, we immediately build it
        self.use_cache = True
        self.cache_built = False
        self.X_cache: torch.Tensor = torch.full((len(self.data_frame),), fill_value=torch.nan)
        self.Y_cache: torch.Tensor = torch.full((len(self.data_frame),), fill_value=torch.nan)
        self.build_cache()

    def __len__(self):
        """
        The length of an `EncodedDs` datasource equals the amount of rows of the original dataframe.

        :return: length of the `EncodedDs`
        """
        return int(self.data_frame.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The getter yields a tuple (X, y), where:
          - `X `is a concatenation of all encoded representations of the row. Size: (B, n_features)
          - `y` is the encoded target. Size: (B, n_features)
          
        :param idx: index of the row to access.
        
        :return: tuple (X, y) with encoded data.
        
        """  # noqa
        if self.use_cache and self.X_cache[idx] is not torch.nan:
            X = self.X_cache[idx, :]
            Y = self.Y_cache[idx]
        else:
            X, Y = self._encode_idxs([idx, ])
            if self.use_cache:
                self.X_cache[idx, :] = X
                self.Y_cache[idx, :] = Y

        return X, Y

    def _encode_idxs(self, idxs: list):
        if not isinstance(idxs, list):
            raise Exception(f"Passed indexes is not an iterable. Check the type! Index: {idxs}")

        X = torch.zeros((len(idxs), self.input_length))
        Y = torch.zeros((len(idxs),))
        for col in self.data_frame:
            if self.encoders.get(col, None):
                kwargs = {}
                if 'dependency_data' in inspect.signature(self.encoders[col].encode).parameters:
                    kwargs['dependency_data'] = {dep: [self.data_frame.iloc[idxs][dep]]
                                                 for dep in self.encoders[col].dependencies}
                if hasattr(self.encoders[col], 'data_window'):
                    cols = [self.target] + [f'{self.target}_timestep_{i}'
                                            for i in range(1, self.encoders[col].data_window)]
                    data = self.data_frame[cols].iloc[idxs].values
                else:
                    cols = [col]
                    data = self.data_frame[cols].iloc[idxs].values.flatten()

                encoded_tensor = self.encoders[col].encode(data, **kwargs)
                if torch.isnan(encoded_tensor).any() or torch.isinf(encoded_tensor).any():
                    raise Exception(f'Encoded tensor: {encoded_tensor} contains nan or inf values, this tensor is \
                                      the encoding of column {col} using {self.encoders[col].__class__}')
                if col != self.target:
                    a, b = self.encoder_spans[col]
                    X[:, a:b] = torch.squeeze(encoded_tensor, dim=list(range(2, len(encoded_tensor.shape))))

                # target post-processing
                else:
                    Y = encoded_tensor

                    if len(encoded_tensor.shape) > 2:
                        Y = encoded_tensor.squeeze()

                    if len(encoded_tensor.shape) < 2:
                        Y = encoded_tensor.unsqueeze(1)

                    # else:
                    #     Y = encoded_tensor.ravel()

        return X, Y

    def get_column_original_data(self, column_name: str) -> pd.Series:
        """
        Gets the original data for any given column of the `EncodedDs`.

        :param column_name: name of the column.
        :return:  A `pd.Series` with the original data stored in the `column_name` column.
        """
        return self.data_frame[column_name]

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        """
        Gets the encoded data for any given column of the `EncodedDs`.

        :param column_name: name of the column.
        :return: A `torch.Tensor` with the encoded data of the `column_name` column.
        """
        if self.use_cache and self.cache_built:
            if column_name == self.target and self.Y_cache is not None:
                return self.Y_cache
            elif self.X_cache is not torch.nan:
                a, b = self.encoder_spans[column_name]
                return self.X_cache[:, a:b]

        kwargs = {}
        if 'dependency_data' in inspect.signature(self.encoders[column_name].encode).parameters:
            deps = [dep for dep in self.encoders[column_name].dependencies if dep in self.data_frame.columns]
            kwargs['dependency_data'] = {dep: self.data_frame[dep] for dep in deps}
        encoded_data = self.encoders[column_name].encode(self.data_frame[column_name], **kwargs)
        if torch.isnan(encoded_data).any() or torch.isinf(encoded_data).any():
            raise Exception(f'Encoded tensor: {encoded_data} contains nan or inf values')

        if not isinstance(encoded_data, torch.Tensor):
            raise Exception(
                f'The encoder: {self.encoders[column_name]} for column: {column_name} does not return a Tensor!')

        if self.use_cache and not self.cache_built:
            if column_name == self.target:
                self.Y_cache = encoded_data
            else:
                a, b = self.encoder_spans[column_name]
                self.X_cache = self.X_cache[:, a:b]

        return encoded_data

    def get_encoded_data(self, include_target: bool = True) -> torch.Tensor:
        """
        Gets all encoded data.

        :param include_target: whether to include the target column in the output or not.
        :return: A `torch.Tensor` with the encoded dataframe.
        """
        encoded_dfs = []
        for col in self.data_frame.columns:
            if (include_target or col != self.target) and self.encoders.get(col, False):
                encoded_dfs.append(self.get_encoded_column_data(col))

        return torch.cat(encoded_dfs, 1)

    def build_cache(self):
        """ This method builds a cache for the entire dataframe provided at initialization. """
        if not self.use_cache:
            raise RuntimeError("Cannot build a cache for EncodedDS with `use_cache` set to False.")

        idxs = list(range(len(self.data_frame)))
        X, Y = self._encode_idxs(idxs)
        self.X_cache = X
        self.Y_cache = Y
        self.cache_built = True

    def clear_cache(self):
        """ Clears the `EncodedDs` cache. """
        self.X_cache = torch.full((len(self.data_frame),), fill_value=torch.nan)
        self.Y_cache = torch.full((len(self.data_frame),), fill_value=torch.nan)
        self.cache_built = False


class ConcatedEncodedDs(EncodedDs):
    """
    `ConcatedEncodedDs` abstracts over multiple encoded datasources (`EncodedDs`) as if they were a single entity.
    """  # noqa

    # TODO: We should probably delete this abstraction, it's not really useful and it adds complexity/overhead
    def __init__(self, encoded_ds_arr: List[EncodedDs]) -> None:
        # @TODO: missing super() call here?
        self.encoded_ds_arr = encoded_ds_arr
        self.encoded_ds_lengths = [len(x) for x in self.encoded_ds_arr]
        self.encoders = self.encoded_ds_arr[0].encoders
        self.encoder_spans = self.encoded_ds_arr[0].encoder_spans
        self.target = self.encoded_ds_arr[0].target
        self.data_frame = pd.concat([x.data_frame for x in self.encoded_ds_arr])

    def __len__(self):
        """
        See `lightwood.data.encoded_ds.EncodedDs.__len__()`.
        """
        # @TODO: behavior here is not intuitive
        return max(0, np.sum(self.encoded_ds_lengths) - 2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        See `lightwood.data.encoded_ds.EncodedDs.__getitem__()`.
        """
        for ds_idx, length in enumerate(self.encoded_ds_lengths):
            if idx - length < 0:
                return self.encoded_ds_arr[ds_idx][idx]
            else:
                idx -= length
        raise StopIteration()

    def get_column_original_data(self, column_name: str) -> pd.Series:
        """
        See `lightwood.data.encoded_ds.EncodedDs.get_column_original_data()`.
        """
        encoded_df_arr = [x.get_column_original_data(column_name) for x in self.encoded_ds_arr]
        return pd.concat(encoded_df_arr)

    def get_encoded_column_data(self, column_name: str) -> torch.Tensor:
        """
        See `lightwood.data.encoded_ds.EncodedDs.get_encoded_column_data()`.
        """
        encoded_df_arr = [x.get_encoded_column_data(column_name) for x in self.encoded_ds_arr]
        return torch.cat(encoded_df_arr, 0)

    def clear_cache(self):
        """
        See `lightwood.data.encoded_ds.EncodedDs.clear_cache()`.
        """
        for ds in self.encoded_ds_arr:
            ds.clear_cache()
