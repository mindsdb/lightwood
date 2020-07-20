import logging

import torch
import torch.nn as nn


def tensor_from_series(series, device, n_dims, pad_value, max_len):
    """
    :param series: list of lists, corresponds to time series: [[x1_1, ..., x1_n], [x2_1, ..., x2_n], ...]
                   the series is zero-padded on each axis so that all dimensions have equal length
    :param device: computing device that PyTorch backend uses
    :param n_dims: will zero-pad dimensions until series_dimensions == n_dims
    :param pad_value: value to pad each dimension in the time series, if needed
    :param max_len: length to pad or truncate each time_series
    :return: series as a tensor ready for model consumption, shape (1, ts_length, n_dims)
    """
    # edge case for when series == "dim 1 data"
    if type(series) != type([]):
        series = [str(series).split(' ')]

    # conversion to float
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

    # timestep padding and truncating
    for i in range(len(float_series)):
        for _ in range(max(0, max_len - len(float_series[i]))):
            float_series[i].append(pad_value)
        float_series[i] = float_series[i][:max_len]

    # dimension padding
    for _ in range(max(0, n_dims - len(float_series))):
        float_series.append([pad_value] * max_len)

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
