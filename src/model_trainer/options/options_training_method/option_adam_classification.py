"""Cross-entropy training hyperparameters for CNN classifiers."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import Adam, Optimizer

from model_trainer.types import TrainingMethodOption

from .utils import (
    collect_predictions_conv1d,
    evaluate_epoch_conv1d,
    format_conv1d_batch,
    train_epoch_conv1d,
)

__all__ = ["_adam_classification"]


def _optimizer_builder(*, model: nn.Module) -> Optimizer:
    return Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def _criterion_builder() -> nn.Module:
    return nn.CrossEntropyLoss()


_adam_classification = TrainingMethodOption(
    name="adam_classification",
    epochs=30,
    batch_size=256,
    optimizer_builder=_optimizer_builder,
    criterion_builder=_criterion_builder,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    device="cpu",
    target_kind="classification",
    batch_formatter=format_conv1d_batch,
    train_epoch_fn=train_epoch_conv1d,
    evaluate_epoch_fn=evaluate_epoch_conv1d,
    prediction_collector=collect_predictions_conv1d,
)
