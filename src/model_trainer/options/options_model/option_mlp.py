"""Simple multilayer perceptron for flattened segment features."""

from __future__ import annotations

from torch import Tensor, nn

from model_trainer.types import ModelOption

__all__ = ["_mlp_classification"]


class _MLP(nn.Module):
    """Two-hidden-layer MLP with lazy input sizing."""

    def __init__(self, *, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128, out_features=output_size),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x)


def _build_mlp_classification(*, output_size: int) -> _MLP:
    return _MLP(output_size=output_size)


_mlp_classification = ModelOption(
    name="mlp_classification",
    model_builder=_build_mlp_classification,
    output_size=9,
    backend="torch",
    target_kind="classification",
)
