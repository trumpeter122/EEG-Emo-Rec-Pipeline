"""Shared training helpers for optimizer/criterion combinations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from model_trainer.types.training_method import BatchFormatter

__all__ = [
    "collect_predictions_conv1d",
    "evaluate_epoch_conv1d",
    "format_conv1d_batch",
    "train_epoch_conv1d",
]


def _ensure_conv1d_shape(batch: torch.Tensor) -> torch.Tensor:
    """Return tensors shaped as ``(batch, channels=1, length)``."""
    if batch.ndim == 2:
        return batch.unsqueeze(1)
    if batch.ndim == 3 and batch.shape[1] == 1:
        return batch
    return batch.reshape(batch.shape[0], 1, -1)


def format_conv1d_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    target_kind: Literal["regression", "classification"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move tensors onto ``device`` and normalize dtypes for Conv1d models."""
    formatted_features = _ensure_conv1d_shape(
        features.to(device=device, dtype=torch.float32),
    )
    if target_kind == "classification":
        formatted_targets = targets.to(device=device, dtype=torch.long)
    else:
        formatted_targets = targets.to(device=device, dtype=torch.float32)
        if formatted_targets.ndim == 1:
            formatted_targets = formatted_targets.unsqueeze(1)
    return formatted_features, formatted_targets


def _classification_mae(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    class_values: torch.Tensor,
) -> torch.Tensor:
    """MAE between predicted and true class center values."""
    pred_indices = torch.argmax(outputs, dim=1)
    pred_values = class_values[pred_indices]
    true_values = class_values[labels]
    return F.l1_loss(
        pred_values.unsqueeze(1),
        true_values.unsqueeze(1),
        reduction="sum",
    )


def train_epoch_conv1d(
    model: nn.Module,
    loader: DataLoader[tuple[np.ndarray, float]],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_kind: Literal["regression", "classification"],
    class_values: torch.Tensor | None,
    batch_formatter: BatchFormatter,
) -> tuple[float, float]:
    """Run one training epoch for Conv1d-compatible datasets."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for features, targets in loader:
        inputs, labels = batch_formatter(
            features,
            targets,
            device,
            target_kind,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        if target_kind == "classification":
            if class_values is None:
                raise RuntimeError(
                    "class_values tensor is required for classification."
                )
            mae = _classification_mae(outputs, labels, class_values)
        else:
            mae = F.l1_loss(outputs, labels, reduction="sum")
        total_mae += float(mae.item())
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_mae = total_mae / max(1, total_samples)
    return avg_loss, avg_mae


def evaluate_epoch_conv1d(
    model: nn.Module,
    loader: DataLoader[tuple[np.ndarray, float]],
    criterion: nn.Module,
    device: torch.device,
    target_kind: Literal["regression", "classification"],
    class_values: torch.Tensor | None,
    batch_formatter: BatchFormatter,
) -> tuple[float, float]:
    """Evaluate a Conv1d-compatible model without gradient tracking."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in loader:
            inputs, labels = batch_formatter(
                features,
                targets,
                device,
                target_kind,
            )
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            if target_kind == "classification":
                if class_values is None:
                    raise RuntimeError(
                        "class_values tensor is required for classification.",
                    )
                mae = _classification_mae(outputs, labels, class_values)
            else:
                mae = F.l1_loss(outputs, labels, reduction="sum")
            total_mae += float(mae.item())
            total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_mae = total_mae / max(1, total_samples)
    return avg_loss, avg_mae


def collect_predictions_conv1d(
    model: nn.Module,
    loader: DataLoader[tuple[np.ndarray, float]],
    device: torch.device,
    target_kind: Literal["regression", "classification"],
    class_values: np.ndarray | None,
    batch_formatter: BatchFormatter,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened predictions/targets for downstream metric reporting."""
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    with torch.no_grad():
        for features, targets in loader:
            inputs, labels = batch_formatter(
                features,
                targets,
                device,
                target_kind,
            )
            outputs = model(inputs)
            if target_kind == "classification":
                pred_indices = torch.argmax(outputs, dim=1, keepdim=True)
                preds.append(pred_indices.detach().cpu().numpy())
                trues.append(labels.detach().cpu().unsqueeze(1).cpu().numpy())
            else:
                preds.append(outputs.detach().cpu().numpy())
                trues.append(labels.detach().cpu().numpy())

    if not preds:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    pred_array = np.concatenate(preds, axis=0).reshape(-1)
    true_array = np.concatenate(trues, axis=0).reshape(-1)
    if target_kind == "classification" and class_values is not None:
        pred_array = class_values[pred_array.astype(int)]
        true_array = class_values[true_array.astype(int)]
    return pred_array, true_array
