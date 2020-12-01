import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def tensor_from_series(series, device, n_dims, pad_value, max_len=None, normalizer=None):
    """
    :param series: list of lists, corresponds to a time series: [[x1_1, ..., x1_n], [x2_1, ..., x2_n], ...]
                   the series will be padded to be (1, ts_length, n_dims)-shaped
    :param device: computing device that PyTorch backend uses
    :param n_dims: number of features in the time series
    :param pad_value: value to pad the time series with, if needed
    :param max_len: number of timesteps in the time_series, automatically determined if it's not passed
    :param normalizer: optional, should have an .encode() method
    :return: series as a tensor ready for model consumption, shape (1, ts_length, n_dims)
    """
    # conversion to float
    if max_len is None:
        max_len = len(series[0]) if isinstance(series, list) else series.shape[1]

    # timestep padding and truncating
    for i in range(len(series)):
        for _ in range(max(0, max_len - len(series[i]))):
            series[i].insert(0, pad_value)
        series[i] = series[i][:max_len]

    if normalizer:
        if isinstance(normalizer, MinMaxNormalizer):
            series = torch.Tensor([normalizer.encode(s)[0] for s in series]).transpose(0, 1)
        else:
            series = torch.Tensor([normalizer.encode(s) for s in series][0])
    else:
        series = torch.Tensor(series).transpose(0, 1)

    # dimension padding
    for _ in range(max(0, n_dims - series.shape[-1])):
        pad = torch.full((series.shape[0], 1), pad_value)
        series = torch.cat([series, pad], dim=-1)

    # remove nans and add batch dimension
    series[torch.isnan(series)] = 0.0
    return series.unsqueeze(0).to(device)


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

    def prepare(self, x):
        X = np.array([j for i in x for j in i]).reshape(-1, 1)
        self.scaler.fit(X)

    def encode(self, y):
        if not isinstance(y[0], list):
            y = np.array(y).reshape(1, -1)
        return self.scaler.transform(y)

    def decode(self, y):
        return self.scaler.inverse_transform(y)


class CatNormalizer:
    def __init__(self, factor=1):
        self.scaler = OneHotEncoder(sparse=False, handle_unknown='ignore')
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
            out.append(self.scaler.transform(i.reshape(-1, 1)))
        return np.array(out)

    def decode(self, y):
        return [[i[0] for i in self.scaler.inverse_transform(o)] for o in y]