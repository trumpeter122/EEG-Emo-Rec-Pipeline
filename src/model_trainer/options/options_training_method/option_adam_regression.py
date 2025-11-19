"""Default regression-oriented training hyperparameters."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Adam, Optimizer

from model_trainer.types import TrainingMethodOption

__all__ = ["_adam_regression"]


def _optimizer_builder(*, model: nn.Module) -> Optimizer:
    return Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def _criterion_builder() -> nn.Module:
    return nn.MSELoss()


_adam_regression = TrainingMethodOption(
    name="adam_regression",
    epochs=10,
    batch_size=64,
    optimizer_builder=_optimizer_builder,
    criterion_builder=_criterion_builder,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    target_kind="regression",
)
