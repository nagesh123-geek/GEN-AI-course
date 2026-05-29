# MINI-GPT CLASS

import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dataset_factory import *
from causal_attention import CausalSelfAttention



class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, seq_len=32):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)

        self.attn = CausalSelfAttention(embed_dim)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device))

        x = tok + pos

        self_attn, causal_attn, x = self.attn(x)

        logits = self.fc(x)

        return logits
