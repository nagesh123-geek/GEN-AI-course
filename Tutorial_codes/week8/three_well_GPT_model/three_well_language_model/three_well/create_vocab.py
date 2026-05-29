
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



data = np.loadtxt("cluster_traj.txt", dtype=str)
set_of_data = set(data)
set_to_list = list(set_of_data)
sorted_list = sorted(set_to_list)

vocab = sorted_list

print(vocab)


string_to_integer = {ch:i for i,ch in enumerate(vocab)} # mapping (stoi)
print(string_to_integer)


# Inverse mapping (itos)
integers_to_strings = {i:ch for ch,i in string_to_integer.items() }

print(integers_to_strings)


tokens = np.array([string_to_integer[ch] for ch in data])
print(tokens)
np.savetxt("integer_tokenized_data.txt", tokens, fmt='%d')

vocab_size = len(vocab)
print(vocab_size)  # 3



import json

# save stoi (string --> index)
with open("vocab_stoi.json", "w") as f:
    json.dump(string_to_integer, f, indent=4)

# save itos (index --> string)
# (convert keys to string because JSON doesn't support int keys well)
itos_str = {str(k): v for k, v in integers_to_strings.items()}

with open("vocab_itos.json", "w") as f:
    json.dump(itos_str, f, indent=4)
    
    
    
   
