import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, in_channels=1, num_features=128):
        super(Simple1DCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output shape: (batch, 64, 1)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_features)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        features = self.encoder(x)         # (batch, 64, 1)
        features = features.squeeze(-1)      # (batch, 64)
        projections = self.projection_head(features)
        return projections
