import torch
import numpy as np
import math

# Load graphs
print("Loading graphs...")
graphs = torch.load("SG256_Full_Cleaned_Graphs.pt")

# Sample into train, test, val split
graphs = np.array(graphs)

# Shuffle graphs up
np.random.shuffle(graphs)

# 15% test 15% val, rest to train
n_vt = math.ceil(0.15 * len(graphs))
n_train = len(graphs) - 2 * n_vt

# sample val and test
print(f"Picking {n_vt} graphs for validataion")
val = graphs[0:n_vt]

print(f"Picking {n_vt} graphs for test")
test = graphs[n_vt:2*n_vt]

print(f"Picking {n_train} graphs for training")
train = graphs[2*n_vt:]

print("Saving test, train, val graphs...")
torch.save(val, "SG256_Full_Cleaned_Graphs_Val.pt")
torch.save(test, "SG256_Full_Cleaned_Graphs_Test.pt")
torch.save(train, "SG256_Full_Cleaned_Graphs_Train.pt")
