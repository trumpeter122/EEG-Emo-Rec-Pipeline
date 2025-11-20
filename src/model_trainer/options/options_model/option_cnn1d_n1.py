"""Baseline 1D CNN model used for both regression and classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNN1D_N1"]


class CNN1D_N1(nn.Module):
    """
    Three-block convolutional encoder feeding a shallow fully-connected head.

    The dense layers rely on ``nn.LazyLinear`` so the model adapts to any
    flattened length produced by the convolutional stack.  When adding new
    architectures, ensure they follow this adaptive pattern rather than
    hardcoding a fixed number of timesteps or channels.
    """

    def __init__(self, *, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.LazyLinear(out_features=64)
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(in_features=16, out_features=output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the CNN encoder followed by the dense classification/regression head."""
        encoded = self.pool1(self.bn1(F.relu(self.conv1(inputs))))
        encoded = self.pool2(self.bn2(F.relu(self.conv2(encoded))))
        encoded = self.pool3(F.relu(self.conv3(encoded)))

        flat = self.flatten(encoded)

        hidden = torch.tanh(self.fc1(flat))
        hidden = self.dropout1(hidden)
        hidden = torch.tanh(self.fc2(hidden))
        hidden = self.dropout2(hidden)
        hidden = F.relu(self.fc3(hidden))
        hidden = self.dropout3(hidden)
        outputs = self.fc4(hidden)
        return outputs
