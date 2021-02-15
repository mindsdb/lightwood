import logging

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TemporalEncBlock, self).__init__()
        self.pad1 = nn.ZeroPad2d((padding, 0))
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)                                           
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ZeroPad2d((padding, 0))
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)                                           
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.pad1, self.conv1, self.bn1, self.relu1,
                                 self.pad2, self.conv2, self.bn2, self.relu2)
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
        print(self.network)

    def forward(self, x):
        return self.network(x)

    def bptt(self, data, criterion, device):
        """This method encodes an input unrolled through time"""
        loss = 0
        next_tensor = data[:, 0, :].unsqueeze(dim=1).transpose(1, 2)  # initial input

        for tensor_i in range(data.shape[1] - 1):
            rand = np.random.randint(2)
            # teach from forward as well as from known tensor alternatively
            if rand == 1:
                next_tensor = self.forward(data[:, tensor_i, :].unsqueeze(dim=1).transpose(1, 2))
            else:
                next_tensor = self.forward(next_tensor.detach())

            loss += criterion(next_tensor, data[:, tensor_i + 1, :].unsqueeze(dim=1).transpose(1, 2))

        return next_tensor, loss


class TemporalDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TemporalDecBlock, self).__init__()
        self.pad2 = nn.ZeroPad2d((padding, 0))
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)                                           
        self.relu2 = nn.ReLU()
        self.pad1 = nn.ZeroPad2d((padding, 0))
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
        for i in range(num_levels-1, -1, -1):
            dilation = 1 if i == 0 else 2 * i      # 2 ** i
            padding = dilation * (kernel_size - 1)
            in_channels = blocks[i]
            out_channels = output_dims if i == 0 else blocks[i-1]
            layers += [TemporalDecBlock(in_channels, out_channels, kernel_size,
                       dilation=dilation, padding=padding)]

        self.network = nn.Sequential(*layers)        

    def forward(self, x):
        return self.network(x)

    def decode(self, data, criterion, device):
        out = self.forward(data).to(device)
        target = torch.roll(data, -1, dims=1)
        return out, criterion(out[:, 0:-1, :], target[:, 0:-1, :])
