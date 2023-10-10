import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

graphs = torch.load('SG256_Full_Cleaned_Graphs_Train.pt') # In the future change this directory to training set
# Special parsing because np test/train/split
graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]
print(f'loaded {len(graphs)} graphs')

# Select specific features for each data (currently mass and redshift)
for graph in graphs:
    graph.x = torch.tensor([[data[0], data[1]] for data in graph.x])
    
valid = True
for graph in graphs:
    if not graph.validate():
        valid = False
print(f'Graphs are valid?: {valid}')
print(f'input shape for graph 0: {graphs[0].x.shape}, output shape: {graphs[0].y.shape}')
if not valid:
    exit(1)

# Create PyTorch dataset
class GraphDataset(Dataset):
    def __init__()


# Model arch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        
        self.lin = torch.nn.Linear(128, 1)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        
        x = self.lin(x)
        
        x = self.leaky(x)

        return x

# Custom loss function to ignore non-zero SM values
# from torchmetrics import MeanAbsolutePercentageError

class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        non_zero_mask = targets != 0
        non_zero_predictions = torch.where(non_zero_mask, predictions, torch.zeros_like(predictions))
        non_zero_targets = torch.where(non_zero_mask, targets, torch.zeros_like(targets))
        loss = (non_zero_predictions - non_zero_targets) ** 2
        return torch.mean(loss)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.00025)
loss_fn = torch.nn.MSELoss()

best_state = None
best_loss = float('inf')

epochs = 100
for epoch in range(1, epochs + 1):
    y_hat = torch.tensor([])
    y = torch.tensor([])

    # Run predictions on all graphs before doing loss
    for graph in graphs:
        data = graph.to(device)
        out = model(data)
        y_hat = torch.cat((y_hat, out))
        y = torch.cat((y, data.y))
    # Filter out zero-mass truths from loss gradient calculations
    # non_zero_mask = y != 0
    # non_zero_predictions = torch.squeeze(y_hat)[non_zero_mask]
    # non_zero_truth = y[non_zero_mask]
    loss = loss_fn(torch.squeeze(y_hat), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = model.state_dict()
    if epoch % 5 == 0:
        print(f"Loss on epoch {epoch}: {loss}")
        
# Save model
torch.save(best_state, 'models/model_10_5_23.pt')
print(f"Best model saved with loss {best_loss}")

