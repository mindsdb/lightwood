import datetime
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def tensor_from_series(series, device, n_dims, pad_value, max_len=None, normalizer=None):
    """
    :param series: list of lists, corresponds to time series: [[x1_1, ..., x1_n], [x2_1, ..., x2_n], ...]
                   the series is zero-padded on each axis so that all dimensions have equal length
    :param device: computing device that PyTorch backend uses
    :param n_dims: will zero-pad dimensions until series_dimensions == n_dims
    :param pad_value: value to pad each dimension in the time series, if needed
    :param max_len: length to pad or truncate each time_series
    :return: series as a tensor ready for model consumption, shape (1, ts_length, n_dims)
    """
    if max_len is None:
        max_len = len(series[0]) if isinstance(series, list) else series.shape[1]

    # conversion to float
    if max_len is None:
        max_len = len(series[0]) if isinstance(series, list) else series.shape[1]

    # timestep padding and truncating
    for i in range(len(series)):
        for _ in range(max(0, max_len - len(series[i]))):
            series[i].append(pad_value)
        series[i] = series[i][:max_len]

    # dimension padding
    for _ in range(max(0, n_dims - len(series))):
        series.append([pad_value] * max_len)

    # normalize and transpose
    if normalizer:
        tensor = torch.Tensor([normalizer.encode(s) for s in series][0])
    else:
        tensor = torch.transpose(torch.Tensor(series), 0, 1)

    # add batch dimension
    return tensor.unsqueeze(0).to(device)


class DecoderRNNNumerical(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNNNumerical, self).__init__()
        self.hidden_size = hidden_size
        self.in_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.in_activation(input.float())
        output, hidden = self.gru(output, hidden)
        output = self.dropout(output)
        output = self.out(output)
        return output, hidden

    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


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

    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class MinMaxNormalizer:
    def __init__(self, factor=1):
        self.scaler = MinMaxScaler()
        self.factor = factor

    def prepare_encoder(self, x):
        X = np.array([j for i in x for j in i]).reshape(-1, 1)
        self.scaler.fit(X)

    def encode(self, y):
        return self.scaler.transform(y)

    def decode(self, y):
        return self.scaler.inverse_transform(y)
