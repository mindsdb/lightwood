import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def tensor_from_series(series, device):
    if type(series) != type([]):
        series = str(series).split(' ')

    float_series = []
    for ele in series:
        try:
            float_series.append(float(ele))
        except:
            logging.warning(f'Weird element encountered in timeseries: {ele} !')
            float_series.append(0)

    return torch.tensor(float_series, dtype=torch.float, device=device).view(-1, 1, 1, 1).float()


class DecoderRNNNumerical(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNNNumerical, self).__init__()
        self.hidden_size = hidden_size
        self.in_activation = nn.ReLU()
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden):
        output = self.in_activation(input.float())
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class EncoderRNNNumerical(nn.Module):
    def __init__(self,  hidden_size):
        input_size = 1
        super(EncoderRNNNumerical, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden):

        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
