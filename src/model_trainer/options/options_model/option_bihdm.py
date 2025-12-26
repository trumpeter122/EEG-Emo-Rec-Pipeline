"""BiHDM-inspired hemispheric discrepancy model."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BiHDM"]


class BiHDM(nn.Module):
    """
    Bi-hemispheric discrepancy model with directional RNN streams.

    The implementation follows the paper's core idea: RNN-based
    representations per hemisphere, subtraction-based discrepancy features,
    high-level RNN summarization, and a classifier head.
    """

    def __init__(
        self,
        *,
        output_size: int,
        feature_dim: int = 5,
        dl: int = 32,
        dg: int = 32,
        do: int = 16,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.dl = dl
        self.dg = dg
        self.do = do

        self.left_gru_h = nn.GRU(
            input_size=feature_dim,
            hidden_size=dl,
            batch_first=True,
        )
        self.right_gru_h = nn.GRU(
            input_size=feature_dim,
            hidden_size=dl,
            batch_first=True,
        )
        self.left_gru_v = nn.GRU(
            input_size=feature_dim,
            hidden_size=dl,
            batch_first=True,
        )
        self.right_gru_v = nn.GRU(
            input_size=feature_dim,
            hidden_size=dl,
            batch_first=True,
        )
        self.high_gru_h = nn.GRU(
            input_size=dl,
            hidden_size=dg,
            batch_first=True,
        )
        self.high_gru_v = nn.GRU(
            input_size=dl,
            hidden_size=dg,
            batch_first=True,
        )
        self.proj_h = nn.Linear(in_features=dg, out_features=do)
        self.proj_v = nn.Linear(in_features=dg, out_features=do)
        self.classifier = nn.Linear(in_features=do * 2, out_features=output_size)

    def _flatten_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            flat = inputs.squeeze(1)
        elif inputs.ndim == 2:
            flat = inputs
        else:
            flat = inputs.view(inputs.shape[0], -1)
        return flat

    def _split_hemispheres(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_electrodes = inputs.shape[1]
        mid = n_electrodes // 2
        if mid == 0:
            raise ValueError("BiHDM requires at least two electrodes.")
        left = inputs[:, :mid, :]
        right = inputs[:, -mid:, :]
        paired = min(left.shape[1], right.shape[1])
        return left[:, :paired, :], right[:, :paired, :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run hemispheric RNNs, discrepancy module, and classifier."""
        flat = self._flatten_inputs(inputs)
        if flat.shape[1] % self.feature_dim != 0:
            raise ValueError(
                "BiHDM expects features divisible by feature_dim="
                f"{self.feature_dim}, got length={flat.shape[1]}.",
            )
        n_electrodes = flat.shape[1] // self.feature_dim
        reshaped = flat.view(flat.shape[0], n_electrodes, self.feature_dim)

        left, right = self._split_hemispheres(reshaped)
        left_h, _ = self.left_gru_h(left)
        right_h, _ = self.right_gru_h(right)

        left_v, _ = self.left_gru_v(torch.flip(left, dims=[1]))
        right_v, _ = self.right_gru_v(torch.flip(right, dims=[1]))

        diff_h = left_h - right_h
        diff_v = left_v - right_v

        _, high_h = self.high_gru_h(diff_h)
        _, high_v = self.high_gru_v(diff_v)

        proj_h = torch.tanh(self.proj_h(high_h.squeeze(0)))
        proj_v = torch.tanh(self.proj_v(high_v.squeeze(0)))
        fused = torch.cat([proj_h, proj_v], dim=1)
        return self.classifier(fused)
