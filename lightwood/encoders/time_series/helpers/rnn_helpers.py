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
        series = normalizer.transform(np.array(series))
    tensor = torch.transpose(torch.tensor(series, dtype=torch.float, device=device), 0, 1)

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

    def fit(self, x):
        X = np.array([j for i in x for j in i]).reshape(-1, 1)
        self.scaler.fit(X)

    def transform(self, y):
        return self.scaler.transform(y)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, y):
        return self.scaler.inverse_transform(y)


class DateNormalizer:
    def __init__(self, factor=1):
        self.fields = ['year', 'month', 'day', 'hour', 'minute', 'second']
        self.constants = {'year': 3000, 'seasonality': 53, 'month': 12,
                          'day': 31, 'hour': 24, 'minute': 60, 'second': 60}
        self.factor = factor

    def normalize_date(self, date):
        norms = []
        for f in self.fields:
            norms.append(getattr(date, f) / self.constants[f])
        return np.array([(np.sin(n), np.cos(n)) for n in norms]).flatten()

    def denormalize_date(self, date):
        arr = []
        for i in range(len(self.fields)):
            sin = date[2*i:2*i+1]
            n = np.rint(np.arcsin(sin) * self.constants[self.fields[i]])
            arr.append(n)
        return datetime.datetime(*arr)

    def transform(self, y):
        """
        :param y: pd.df column with datetime objects
        :return: numpy array of shape (N, 2*len(self.fields))
        """
        tmp = y.map(self.normalize_date).values
        out = np.zeros((tmp.shape[0], tmp[0].shape[0]))
        for i in range(out.shape[0]):
            out[i, :] = tmp[i]

        return out

    def inverse_transform(self, y):
        """
        :param y: numpy array of shape (N, 2*len(self.fields))
        :return: array with datetime objects, shape (N,)
        """
        out = np.apply_along_axis(self.denormalize_date, 1, y)
        return out
