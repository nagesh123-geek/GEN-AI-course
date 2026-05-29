import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dataset_factory import *
from causal_attention import CausalSelfAttention
from gpt import *

import json


# Set device (This is important as all computations are done and saved on device)
DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Leo,  We are using : " , DEVICE, "for computation")




with open("vocab_stoi.json", "r") as f:
    string_to_integer = json.load(f)
    
with open("vocab_itos.json", "r") as f:
    itos_str = json.load(f)

# convert keys back to int
integers_to_strings = {int(k): v for k, v in itos_str.items()}

vocab_size = len(string_to_integer)


# dataset + loader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# model
model = MiniGPT(vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



loss_history = []

for epoch in range(1000):
    total_loss = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            yb.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}",flush=True)


np.savetxt("training_loss.txt", loss_history)
print("Saved loss to training_loss.txt")


torch.save(model.state_dict(), "gpt_cluster_model.pth")
print("Model saved")




