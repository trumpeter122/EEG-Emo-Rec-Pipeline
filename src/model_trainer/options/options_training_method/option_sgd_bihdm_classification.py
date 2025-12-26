"""SGD training configuration aligned with the BiHDM paper."""

from __future__ import annotations

import torch.nn as nn
from torch.optim import SGD, Optimizer

from model_trainer.types import TrainingMethodOption

from .utils import (
    collect_predictions_conv1d,
    evaluate_epoch_conv1d,
    format_conv1d_batch,
    train_epoch_conv1d,
)

__all__ = ["_sgd_bihdm_classification"]


def _optimizer_builder(*, model: nn.Module) -> Optimizer:
    return SGD(model.parameters(), lr=3e-3, momentum=0.9, weight_decay=0.95)


def _criterion_builder() -> nn.Module:
    return nn.CrossEntropyLoss()


_sgd_bihdm_classification = TrainingMethodOption(
    name="sgd_bihdm_classification",
    epochs=30,
    batch_size=200,
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
