import torch
import numpy as np
from torch_geometric.data import Data

# Load pruned graphs with baryonic info
print("Loading graphs...")
graphs = torch.load("SG256_Full_Graphs.pt")

# Special parsing because np test/train/split
if not isinstance(graphs, list) or not isinstance(graphs[0], Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

# Remove halos with <= 0 stellar mass
cleaned_graphs = []
print(f"Cleaning {len(graphs)} graphs...")
for graph in graphs:
    # find indices without -1 for stellar mass
    valid_halo_idxs = np.where(graph.y > 0)[0]
    valid_halo_idxs = torch.from_numpy(valid_halo_idxs)

    # create subgraph with those halos
    cleaned_graph = graph.subgraph(valid_halo_idxs)

    # alert if any halos were cut
    # invalid = len(graph.y) - len(valid_halo_idxs) 
    # if invalid > 0:
    #    print(f"{invalid} invalid halos found for this graph!")

    # only append if there are any halos left
    if len(cleaned_graph.y) > 0:
        cleaned_graphs.append(cleaned_graph)

print("Saving cleaned graphs...")
torch.save(cleaned_graphs, "SG256_Full_ONLY_SM.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
