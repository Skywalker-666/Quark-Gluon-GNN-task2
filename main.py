import os
import gc
import torch
from torch_geometric.loader import DataLoader
from src.data_utils import load_file_as_graphs
from src.models import GCNNet, GATNet

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./data/" # Change to your directory
files = sorted([f for f in os.listdir(data_path) if f.endswith(".npz")])

gcn = GCNNet().to(device)
gat = GATNet().to(device)

opt_gcn = torch.optim.Adam(gcn.parameters(), lr=0.001)
opt_gat = torch.optim.Adam(gat.parameters(), lr=0.001)

def run_epoch(model, loader, optimizer, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct = 0, 0
    
    for data in loader:
        data = data.to(device)
        if is_train: optimizer.zero_grad()
        
        out = model(data)
        loss = torch.nn.functional.cross_entropy(out, data.y)
        
        if is_train:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(dim=1) == data.y).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)

# Training Loop
for file in files:
    print(f"Processing: {file}")
    dataset = load_file_as_graphs(os.path.join(data_path, file))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(2):
        l_gcn, a_gcn = run_epoch(gcn, loader, opt_gcn)
        l_gat, a_gat = run_epoch(gat, loader, opt_gat)
        print(f"Ep {epoch} | GCN Acc: {a_gcn:.4f} | GAT Acc: {a_gat:.4f}")

    del dataset, loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
