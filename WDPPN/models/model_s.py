# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:39:52 2025

@author: fangwenji
"""

import torch
import torch.nn as nn
from network_module import res1D, WaveletDenoiseNet, FlashAttenBlock, ResidualECA1dBlock

class model_s(nn.Module):
    def __init__(self):
        super(model_s, self).__init__()

        self.cwt_denoise = WaveletDenoiseNet(0.01, 0.1, 0.01, 126, 6)

        self.res1 = nn.Sequential(
            ResidualECA1dBlock(3, 8, 7, 1, 3), 
            ResidualECA1dBlock(8, 8, 7, 1, 3), 
            FlashAttenBlock(8, 6000, 4)
        )
        self.res2 = nn.Sequential(
            ResidualECA1dBlock(8, 16, 5, 1, 2), 
            ResidualECA1dBlock(16, 16, 5, 1, 2), 
            FlashAttenBlock(16, 3000, 4)
        )
        self.res3 = nn.Sequential(
            ResidualECA1dBlock(16, 32, 3), 
            ResidualECA1dBlock(32, 32, 3), 
            FlashAttenBlock(32, 1500, 4)
        )
        self.res4 = nn.Sequential(
            ResidualECA1dBlock(32, 64, 3), 
            ResidualECA1dBlock(64, 64, 3), 
            FlashAttenBlock(64, 750, 4)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.res5 = nn.Sequential(
            ResidualECA1dBlock(64, 128, 3, 1), 
            ResidualECA1dBlock(128, 128, 3, 1), 
            FlashAttenBlock(128, 375, 4)
        )

        self.convup1 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv1d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.res1_up = nn.Sequential(
            res1D(128, 64, 3, 1), 
            res1D(64, 64, 3, 1)
        )
        self.convup2 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv1d(64, 32, kernel_size=3, padding=1), 
            nn.BatchNorm1d(32), nn.ReLU()
        )
        self.res2_up = nn.Sequential(
            res1D(64, 32, 3, 1), 
            res1D(32, 32, 3, 1)
        )
        self.convup3 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv1d(32, 16, kernel_size=5, padding=2), 
            nn.BatchNorm1d(16), nn.ReLU()
        )
        self.res3_up = nn.Sequential(
            res1D(32, 16, 5, 2), 
            res1D(16, 16, 5, 2)
        )
        self.convup4 = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv1d(16, 8, kernel_size=7, padding=3), 
            nn.BatchNorm1d(8), nn.ReLU()
        )
        self.res4_up = nn.Sequential(
            res1D(16, 8, 7, 3), 
            res1D(8, 8, 7, 3)
        )
        self.convup5 = nn.Sequential(nn.Conv1d(8, 1, kernel_size=1, padding=0))

    def forward(self, x):
        y = self.cwt_denoise(x)

        z1 = self.res1(y)
        z1_ds = self.maxpool(z1)
        z2 = self.res2(z1_ds)
        z2_ds = self.maxpool(z2)
        z3 = self.res3(z2_ds)
        z3_ds = self.maxpool(z3)
        z4 = self.res4(z3_ds)
        z4_ds = self.maxpool(z4)

        p1 = self.res5(z4_ds)

        p1 = self.convup1(p1)
        p1 = torch.cat((p1, z4), dim=1)
        p1 = self.res1_up(p1)

        p1 = self.convup2(p1)
        p1 = torch.cat((p1, z3), dim=1)
        p1 = self.res2_up(p1)

        p1 = self.convup3(p1)
        p1 = torch.cat((p1, z2), dim=1)
        p1 = self.res3_up(p1)

        p1 = self.convup4(p1)
        p1 = torch.cat((p1, z1), dim=1)
        p1 = self.res4_up(p1)

        p1 = self.convup5(p1)

        return p1, y