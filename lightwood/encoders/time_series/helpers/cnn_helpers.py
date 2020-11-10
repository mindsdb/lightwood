import logging

import numpy as np 
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class TemporalEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TemporalEncBlock, self).__init__()
        self.pad1 = nn.ZeroPad2d((padding,0))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,)
        self.bn1 = nn.BatchNorm1d(out_channels)                                           
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ZeroPad2d((padding,0))
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)                                           
        self.relu2 = nn.ReLU()
        self.downsample = nn.MaxPool1d(2,2) 

        self.net = nn.Sequential(self.pad1, self.conv1, self.bn1, self.relu1,
                                 self.pad2, self.conv2, self.bn2, self.relu2, self.downsample) 
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None        

    def forward(self, x):
        out = self.net(x)
        # Residual connection:
        res = x if self.shortcut is None else self.shortcut(x)
        res = F.interpolate(res, size=out.size(2), mode='linear', align_corners=True)   
        return self.relu1(out + res)

class EncoderCNNts(nn.Module):
    def __init__(self, input_dims, blocks, kernel_size):
        super(EncoderCNNts, self).__init__()
        layers = []
        num_levels = len(blocks)
        for i in range(num_levels):
            dilation = 1 if i == 0 else 2 * i      # 2 ** i
            padding = dilation * (kernel_size - 1)
            in_channels = input_dims if i == 0 else blocks[i-1]
            out_channels = blocks[i]
            layers += [TemporalEncBlock(in_channels, out_channels, kernel_size,
                                            dilation=dilation, padding=padding)]

        self.network = nn.Sequential(*layers)        

    def forward(self, x):
        return self.network(x)


class TemporalDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TemporalDecBlock, self).__init__()
        self.pad2 = nn.ZeroPad2d((padding,0))
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)                                           
        self.relu2 = nn.ReLU()
        self.pad1 = nn.ZeroPad2d((padding,0))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)                                           
        self.relu1 = nn.ReLU()

        self.net = nn.Sequential(self.pad2, self.conv2, self.bn2, self.relu2,
                                 self.pad1, self.conv1, self.bn1, self.relu1)                              

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        upsampled = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        out = self.net(upsampled)
        res = x if self.shortcut is None else self.shortcut(x)
        res = F.interpolate(res, size=out.size(2), mode='linear', align_corners=True)
        return self.relu1(out + res)

class DecoderCNNts(nn.Module):
    def __init__(self, kernel_size, output_dims, blocks):
        super(DecoderCNNts, self).__init__()
        layers = []
        num_levels = len(blocks)
        for i in range(num_levels-1,-1,-1):
            dilation = 1 if i == 0 else 2 * i      # 2 ** i
            padding = dilation * (kernel_size - 1)
            in_channels = blocks[i]
            out_channels = output_dims if i == 0 else blocks[i-1]
            layers += [TemporalDecBlock(in_channels, out_channels, kernel_size,
                            dilation=dilation, padding=padding)]

        self.network = nn.Sequential(*layers)        

    def forward(self, x):
        return self.network(x)


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
    # edge case for when series == "dim 1 data", outputs shape (1, ts_length)
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


def simple_data_generator(length, dims):
    data = [[0 for x in range(length)] for x in range(dims)]
    for i in range(dims):
        for j in range(length):
            data[i][j] = '%s'%(20*i+j)

    return [data]

def nonlin_data_generator(length, dims):
    data = [[0 for x in range(length)] for x in range(dims)]
    for i in range(dims):
        for j in range(length):
            data[i][j] = '%s'%(j**3+j**2+i)

    return [data]

def random_data_generator(length, dims):
    data = [[0 for x in range(length)] for x in range(dims)]
    for i in range(dims):
        for j in range(length):
            data[i][j] = '%s'%(np.random.randint(0,100))

    return [data]

