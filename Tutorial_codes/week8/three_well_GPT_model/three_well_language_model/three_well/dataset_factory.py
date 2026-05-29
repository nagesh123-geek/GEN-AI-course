# Create dataset
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



tokens = np.loadtxt("integer_tokenized_data.txt")

SEQ_LEN = 32

def create_dataset(tokens):
    X, Y = [], []
    for i in range(len(tokens) - SEQ_LEN):
        X.append(tokens[i:i+SEQ_LEN])
        Y.append(tokens[i+1:i+SEQ_LEN+1])
    return np.array(X), np.array(Y)

X, Y = create_dataset(tokens)



# Convert X and Y to tensors for computational ease

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)


dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# iterate
#for xb, yb in loader:
#    print(xb.shape, yb.shape)
#    break




