import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob 
import os 
import random 
import pickle
import torch 
import torch.nn as nn
from torch.nn import functional as F




class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input-->(batch, time_step, vocab)
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out
        
        
        



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out
        
        


class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)




class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x




class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size = block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm, different from actual transformer
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        # print(index.shape)
        B, T = index.shape
        tok_emb = self.token_embedding_table(index) # (B,T,V), B--->Batch T--->Time step, V--->vocab size
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb # (B,T,V)
        x = self.blocks(x) # (B,T,V)
        x = self.ln_f(x) # (B,T,V)
        logits = self.lm_head(x) # (B,T,V)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits.shape, targets)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            # croping
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            # last time step
            logits = logits[:, -1, :] #(B, V)
            probs = F.softmax(logits, dim=-1) # (B, V)
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index





