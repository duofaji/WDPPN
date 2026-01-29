# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 14:32:30 2025

@author: fangwenji
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

#%% CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#%%wavelet_denoising
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
    
class WaveletDenoisingNet(nn.Module):
    def __init__(self, in_channels=3):
        super(WaveletDenoisingNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 8, kernel_size=3, stride=2, padding=1), 
            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(16, 16, kernel_size=3, stride=2, padding=1)
        )
        
        self.threshold_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        features = self.feature_extractor(x)
        thresh = self.threshold_conv(features)
        thresh_up = F.interpolate(thresh, size=(H, W), mode='bilinear', align_corners=False)
        
        return thresh_up

class WaveletDenoiseNet(nn.Module):
    def __init__(self, dt, dj, s0, j1, param):
        super(WaveletDenoiseNet, self).__init__()
        self.dt = dt
        self.dj = dj
        self.s0 = s0
        self.j1 = j1
        self.param = param
        
        self.threshold_net = WaveletDenoisingNet(in_channels=3)

    def daughter_wavelet_torch(self, k, scale, parameter):
        k0 = parameter
        n = k.shape[-1]
        daughter = torch.sqrt(scale * k[1]) * (torch.pi ** -0.25) * torch.sqrt(torch.tensor(n, dtype=torch.float32)) \
                   * torch.exp(-(scale * k - k0) ** 2 / 2) * (k > 0)
        daughter = daughter * (k > 0)
        s2p_factor = (4 * torch.pi) / (k0 + torch.sqrt(torch.tensor(2 + k0**2)))
        return daughter, s2p_factor

    def cwt(self, X):
        batch_size, channels, n1 = X.shape
        x = X - X.mean(dim=-1, keepdim=True)

        fix_n1_half = n1 // 2
        positive_k = torch.arange(1, fix_n1_half + 1, dtype=torch.float32, device=X.device) * (2 * torch.pi) / (n1 * self.dt)
        negative_k = -positive_k[:(n1 - 1) // 2].flip(0) if (n1 - 1) // 2 > 0 else torch.tensor([], device=X.device)
        k = torch.cat([torch.tensor([0.0], device=X.device), positive_k, negative_k])

        ft = torch.fft.fft(x, dim=-1)
        scale = self.s0 * 2 ** ((torch.arange(self.j1 + 1, dtype=torch.float32, device=X.device)) * self.dj)
        scale = scale.view(1, 1, -1, 1)

        wt_coef = torch.zeros((batch_size, channels, self.j1 + 1, n1), dtype=torch.complex64, device=X.device)

        for kkk in range(self.j1 + 1):
            daughter, s2p_factor = self.daughter_wavelet_torch(k, scale[:, :, kkk, :], self.param)
            wt_coef[:, :, kkk, :] = torch.fft.ifft(ft * daughter, dim=-1)

        period = s2p_factor * scale.squeeze()
        return wt_coef, period, scale.squeeze(), k

    def icwt(self, wt_coef, scale, k):
        batch_size, channels, num_scales, n1 = wt_coef.shape
        WT_Coef = torch.real(wt_coef)

        scale = scale.view(-1, 1)
        ss = scale.expand(num_scales, n1)

        INDEX = torch.sum(WT_Coef / torch.sqrt(ss), dim=2)

        Windex = torch.zeros((batch_size, channels, num_scales), dtype=torch.complex64, device=wt_coef.device)
        for kkk in range(num_scales):
            daughter, _ = self.daughter_wavelet_torch(k, scale[kkk].view(1, 1, 1), self.param)
            Windex[:, :, kkk] = (1 / n1) * torch.sum(daughter, dim=-1)

        RealWindex = torch.real(Windex).view(batch_size, channels, num_scales, 1)
        C = torch.sum(RealWindex / torch.sqrt(scale), dim=2)

        Xrec = (1 / C) * INDEX
        return Xrec

    def scad_threshold(self, wt_coef, thresholds, a: float = 3.7):
        magnitude = torch.abs(wt_coef)
        phase = torch.angle(wt_coef)
        
        sign = torch.sign(magnitude)
        
        cond1 = magnitude <= thresholds
        cond2 = (magnitude > thresholds) & (magnitude <= 2 * thresholds)
        cond3 = (magnitude > 2 * thresholds) & (magnitude <= a * thresholds)
        
        out1 = torch.zeros_like(magnitude)
        out2 = sign * (magnitude - thresholds)
        out3 = sign * (((a - 1) * magnitude - a * thresholds) / (a - 2))
        out4 = magnitude

        new_magnitude = torch.where(
            cond1, out1,
            torch.where(cond2, out2,
            torch.where(cond3, out3, out4))
        )
        
        denoised_coef = new_magnitude * torch.exp(1j * phase)
        return denoised_coef

    def forward(self, X):
        self.wt_coef, period, scale, k = self.cwt(X)

        magnitude = torch.abs(self.wt_coef)
        self.thresholds = self.threshold_net(magnitude)
        self.thresholds = torch.abs(self.thresholds)
        
        wt_coef_denoised = self.scad_threshold(self.wt_coef, self.thresholds)
        Xrec = self.icwt(wt_coef_denoised, scale, k)
        return Xrec
    