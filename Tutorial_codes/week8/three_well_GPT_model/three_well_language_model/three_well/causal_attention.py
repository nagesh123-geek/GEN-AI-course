# Create dataset
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super().__init__()
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** -0.5

        # register causal mask once
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn = Q @ K.transpose(-2, -1) * self.scale

        # self attention (no mask)
        self_attn = F.softmax(attn, dim=-1)

        # use precomputed mask
        causal_mask = self.mask[:T, :T]
        masked_attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        causal_attn = F.softmax(masked_attn, dim=-1)

        out = causal_attn @ V

        return self_attn, causal_attn, out
        
        




