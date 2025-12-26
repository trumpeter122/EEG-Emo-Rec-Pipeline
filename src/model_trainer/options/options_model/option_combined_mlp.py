"""Combined MLP ensemble for wavelet-coefficient classification."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CombinedWaveletMLP"]


class _FirstLevelMLP(nn.Module):
    """Single-hidden-layer MLP used in each first-level branch."""

    def __init__(self, *, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(self.fc1(self.flatten(inputs)))
        return self.fc2(hidden)


class CombinedWaveletMLP(nn.Module):
    """
    Two-stage combined MLP matching the paper's ensemble topology.

    Three first-level MLPs feed a second-level MLP that combines their outputs.
    """

    def __init__(
        self,
        *,
        output_size: int,
        first_hidden: int = 20,
        second_hidden: int = 25,
        branches: int = 3,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                _FirstLevelMLP(output_size=output_size, hidden_size=first_hidden)
                for _ in range(branches)
            ]
        )
        self.combine_fc1 = nn.Linear(
            in_features=output_size * branches,
            out_features=second_hidden,
        )
        self.combine_fc2 = nn.Linear(
            in_features=second_hidden,
            out_features=output_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(inputs) for branch in self.branches]
        combined = torch.cat(branch_outputs, dim=1)
        combined = torch.sigmoid(self.combine_fc1(combined))
        return self.combine_fc2(combined)
