import logging

import torch
import torch.nn as nn
import numpy as np


def float_matrix_from_strlist(series):
    # edge case for when series == "dim 1 data"
    if isinstance(series, str):
        series = [str(series).split(' ')]
    if isinstance(series[0], str):
        series = [str(s).split(' ') for s in series]

    float_series = []
    for dimn in series:
        dimn_series = []
        for ele in dimn:
            try:
                dimn_series.append(float(ele))
            except Exception as _:
                logging.warning(f'Weird element encountered in timeseries: {ele} !')
                dimn_series.append(0)
        float_series.append(dimn_series)
    return float_series


def tensor_from_series(series, device, n_dims, pad_value, max_len, normalizer=None):
    """
    :param series: list of lists, corresponds to time series: [[x1_1, ..., x1_n], [x2_1, ..., x2_n], ...]
                   the series is zero-padded on each axis so that all dimensions have equal length
    :param device: computing device that PyTorch backend uses
    :param n_dims: will zero-pad dimensions until series_dimensions == n_dims
    :param pad_value: value to pad each dimension in the time series, if needed
    :param max_len: length to pad or truncate each time_series
    :return: series as a tensor ready for model consumption, shape (1, ts_length, n_dims)
    """
    # conversion to float
    float_series = float_matrix_from_strlist(series)

    # timestep padding and truncating
    for i in range(len(float_series)):
        for _ in range(max(0, max_len - len(float_series[i]))):
            float_series[i].append(pad_value)
        float_series[i] = float_series[i][:max_len]

    # dimension padding
    for _ in range(max(0, n_dims - len(float_series))):
        float_series.append([pad_value] * max_len)

    # normalize and transpose
    if normalizer:
        float_series = normalizer.transform(np.array(float_series))
    tensor = torch.transpose(torch.tensor(float_series, dtype=torch.float, device=device), 0, 1)

    # add batch dimension
    return tensor.view(-1, max_len, n_dims)


class DecoderRNNNumerical(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNNNumerical, self).__init__()
        self.hidden_size = hidden_size
        self.in_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.in_activation(input.float())
        output, hidden = self.gru(output, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderRNNNumerical(nn.Module):
    def __init__(self,  input_size, hidden_size):
        super(EncoderRNNNumerical, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class TanhNormalizer:
    """ Ref: https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python """
    def __init__(self, factor=10):
        self.mu = None
        self.std = None
        self.factor = factor

    def fit(self, x):
        x = float_matrix_from_strlist(x)

        self.mu = np.mean(x, dtype=np.float64)
        self.std = np.std(x, dtype=np.float64)  # might .mul() by constant for better generalization

    def transform(self, x):
        output = 0.5 * (np.tanh(0.01 * (x - self.mu) / self.std) + 1)
        eps = np.finfo(float).eps
        output = np.where(output == 0, eps, output)  # bound encodings == 0 or 1 to avoid error with arctanh
        output = np.where(output == 1, 1.0-eps, output)
        return output * self.factor

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        return (self.mu + (100 * self.std * np.arctanh(2*(y / self.factor) - 1)))

    def inverse_transform_tensor(self, y):
        return (self.mu + (100 * self.std * torch.atanh(2*(y / self.factor) - 1)))
