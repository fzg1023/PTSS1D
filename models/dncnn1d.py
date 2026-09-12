from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
class DnCNN1D(nn.Module):
    def __init__(self, in_channels=1, num_features=32, num_layers=18):
        super(DnCNN1D, self).__init__()
        layers = []

        layers.append(nn.Conv1d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(num_features, in_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)
