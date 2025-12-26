"""Channel-frequency 2D CNN for review-style EEG replication."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CFCNN"]


class CFCNN(nn.Module):
    """
    Channel-frequency CNN over per-channel band features.

    This mirrors the review paper's channel-frequency CNN idea by reshaping
    flattened band features into a (channels x bands) map per sample.
    """

    def __init__(self, *, output_size: int, bands: int = 5) -> None:
        super().__init__()
        self.bands = bands
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)

    def _reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            flat = inputs.squeeze(1)
        elif inputs.ndim == 2:
            flat = inputs
        else:
            flat = inputs.view(inputs.shape[0], -1)

        if flat.ndim == 1:
            flat = flat.unsqueeze(0)

        length = flat.shape[1]
        if self.bands <= 0 or length % self.bands != 0:
            return flat.view(flat.shape[0], 1, 1, -1)

        channels = length // self.bands
        return flat.view(flat.shape[0], 1, channels, self.bands)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run channel-frequency convolutional encoder + classifier."""
        mapped = self._reshape_inputs(inputs)
        encoded = F.relu(self.bn1(self.conv1(mapped)))
        encoded = F.relu(self.bn2(self.conv2(encoded)))
        flat = self.flatten(encoded)
        hidden = torch.tanh(self.fc1(flat))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)
