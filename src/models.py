import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCNNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x)

class GATNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.conv1 = GATConv(in_channels, 64, heads=4)
        self.conv2 = GATConv(64*4, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x)
