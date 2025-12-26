"""Single-hidden-layer MLP for wavelet K-means probability features."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["KMeansWaveletMLP"]


class KMeansWaveletMLP(nn.Module):
    """MLPNN with one hidden layer (paper specifies 5 hidden neurons)."""

    def __init__(self, *, output_size: int, hidden_size: int = 5) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(self.fc1(self.flatten(inputs)))
        return self.fc2(hidden)
