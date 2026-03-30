import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def build_graph(particles, label, k=8):
    # Remove zero-padding (PT > 0)
    mask = particles[:, 0] > 0
    particles = particles[mask]
    
    if len(particles) < 5:
        return None 

    # Normalize PT: log(1 + x)
    particles[:, 0] = np.log(particles[:, 0] + 1e-6)
    
    x = torch.tensor(particles[:, :3], dtype=torch.float32) # [PT, eta, phi]
    pos = x[:, 1:3]  # Use eta, phi for spatial KNN
    
    edge_index = knn_graph(pos, k=k)
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

def load_file_as_graphs(file_path, max_samples=8000):
    data = np.load(file_path)
    X, y = data['X'], data['y']
    
    dataset = []
    for i in range(min(len(X), max_samples)):
        graph = build_graph(X[i], y[i])
        if graph is not None:
            dataset.append(graph)
    return dataset
