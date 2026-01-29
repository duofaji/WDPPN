# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 14:46:36 2025

@author: fangwenji
"""

import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func
from safari.src.models.sequence.hyena import HyenaOperator

class FlashAttenBlock(nn.Module):
    def __init__(self, dim, length, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.hyena = HyenaOperator(
            d_model=dim,
            l_max=length,
            filter_order=64,
            order=2, 
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.pos_embed = nn.Parameter(torch.zeros(1,length,dim))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        
        batch, seq_len, dim = x.shape
        
        x = x + self.pos_embed
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.half()
        attn_out = flash_attn_qkvpacked_func(qkv).to(torch.float32)
        attn_out1 = self.out_proj(attn_out.view(batch,seq_len,dim))
        
        hyena_out = self.hyena(x)

        fused = self.alpha*hyena_out + (1-self.alpha)*attn_out1
        
        # FFN
        out = self.norm1(x+fused)
        
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out.permute(0,2,1)