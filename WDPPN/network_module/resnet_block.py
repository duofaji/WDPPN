# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:00:58 2025

@author: fangwenji
"""

import torch
import torch.nn as nn

class res2D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super(res2D, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
                                       
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                                nn.BatchNorm2d(out_channel))
        else:
            self.shortcut = nn.Identity()
           
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        
        return out
    
class res1D(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super(res1D, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm1d(out_channel)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm1d(out_channel)
                                       
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                                nn.Conv1d(in_channel, out_channel, kernel_size=1),
                                nn.BatchNorm1d(out_channel))
        else:
            self.shortcut = nn.Identity()
           
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        
        return out
    