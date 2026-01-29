# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:01:55 2025

@author: fangwenji
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_eca_kernel_size(num_channels, gamma=2, b=1):
    """
    Compute adaptive kernel size for ECA: k = |(log2(C)/b + gamma)| odd.
    """
    t = int(abs(((torch.log2(torch.tensor(num_channels, dtype=torch.float32)) + b) / gamma).item()))
    k = t if t % 2 else t + 1
    return k if k > 0 else 1


class ECA1D(nn.Module):
    """
    Efficient Channel Attention (ECA) for 1D signals.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
        # compute adaptive kernel size
        k = get_eca_kernel_size(channels, gamma, b)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        # global average pooling: [B, C, 1]
        y = F.adaptive_avg_pool1d(x, 1)
        # [B, 1, C]
        y = y.permute(0, 2, 1)
        # conv1d along channel dimension
        y = self.conv(y)
        # [B, 1, C] -> [B, C, 1]
        y = y.permute(0, 2, 1)
        weights = self.sigmoid(y)
        # scale x
        return x * weights


class ResidualECA1dBlock(nn.Module):
    """
    1D Residual Block with ECA attention.
    Structure: Conv1d -> BN -> ReLU -> Conv1d -> BN -> ECA -> add Residual -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_downsample=False, gamma=2, b=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = ECA1D(out_channels, gamma=gamma, b=b)
        self.relu = nn.ReLU(inplace=True)
        self.use_downsample = use_downsample
        if use_downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out