"""Option definitions describing how datasets are consumed during training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from config.option_utils import (
    CriterionBuilder,
    OptimizerBuilder,
    _callable_path,
)

BatchFormatter = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.device,
        Literal["regression", "classification"],
    ],
    tuple[torch.Tensor, torch.Tensor],
]
TrainEpochFn = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        Optimizer,
        nn.Module,
        torch.device,
        Literal["regression", "classification"],
        torch.Tensor | None,
        BatchFormatter,
    ],
    tuple[float, float],
]
EvaluateEpochFn = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        nn.Module,
        torch.device,
        Literal["regression", "classification"],
        torch.Tensor | None,
        BatchFormatter,
    ],
    tuple[float, float],
]
PredictionCollector = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        torch.device,
        Literal["regression", "classification"],
        np.ndarray | None,
        BatchFormatter,
    ],
    tuple[np.ndarray, np.ndarray],
]

__all__ = [
    "BatchFormatter",
    "EvaluateEpochFn",
    "PredictionCollector",
    "TrainEpochFn",
    "TrainingMethodOption",
]


@dataclass(slots=True)
class TrainingMethodOption:
    """
    Optimizer/criterion configuration describing how datasets are consumed.

    - The dataclass bundles epoch count, batch sizing, DataLoader knobs, and
      builders that instantiate the actual optimizer + loss modules.
    - Each option is tied to a ``target_kind`` so incompatible dataset variants
      are rejected before training begins.
    """

    name: str
    epochs: int
    batch_size: int
    optimizer_builder: OptimizerBuilder
    criterion_builder: CriterionBuilder
    num_workers: int
    pin_memory: bool
    drop_last: bool
    device: Literal["cpu", "cuda"]
    target_kind: Literal["regression", "classification"]
    batch_formatter: BatchFormatter
    train_epoch_fn: TrainEpochFn
    evaluate_epoch_fn: EvaluateEpochFn
    prediction_collector: PredictionCollector

    def __post_init__(self) -> None:
        """Validate input parameters."""
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    def build_optimizer(self, *, model: nn.Module) -> Optimizer:
        """
        Instantiate the optimizer for ``model``.

        Args:
        ----
            model: The neural network model.

        Returns:
        -------
            An instantiated optimizer.
        """
        return self.optimizer_builder(model=model)

    def build_criterion(self) -> nn.Module:
        """
        Instantiate the configured loss function.

        Returns:
        -------
            An instantiated loss function.
        """
        return self.criterion_builder()

    def build_dataloader(
        self,
        *,
        dataset: Dataset[tuple[np.ndarray, float]],
        shuffle: bool,
    ) -> DataLoader[tuple[np.ndarray, float]]:
        """
        Create a ``DataLoader`` suitable for the configured training regime.

        Args:
        ----
            dataset: The dataset to load.
            shuffle: Whether to shuffle the data.

        Returns:
        -------
            A DataLoader instance.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def to_params(self) -> dict[str, Any]:
        """Serialize method hyperparameters."""
        return {
            "name": self.name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer_builder": _callable_path(self.optimizer_builder),
            "criterion_builder": _callable_path(self.criterion_builder),
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "target_kind": self.target_kind,
            "batch_formatter": _callable_path(self.batch_formatter),
            "train_epoch_fn": _callable_path(self.train_epoch_fn),
            "evaluate_epoch_fn": _callable_path(self.evaluate_epoch_fn),
            "prediction_collector": _callable_path(self.prediction_collector),
        }
