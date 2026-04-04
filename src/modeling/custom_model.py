import torch
import torch.nn as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TemporalRiskMLP(nn.Module):
    """
    A simple Multilayer Perceptron for brokerage drift risk scoring using PyTorch.
    Input size corresponds to the number of SENA temporal features.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super(TemporalRiskMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_pytorch_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=64, learning_rate=0.005):
    """
    Helper function to wrap PyTorch training loops for consistent integration with TimeSeriesSplit.
    """
    # Convert numpy arrays/pandas dfs to torch tensors
    X_tr = torch.tensor(X_train.values, dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_te = torch.tensor(X_test.values, dtype=torch.float32)
    
    # Create DataLoader
    train_data = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = TemporalRiskMLP(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for __ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation Mode
    model.eval()
    with torch.no_grad():
        preds = model(X_te).squeeze().numpy()
        
    return preds
