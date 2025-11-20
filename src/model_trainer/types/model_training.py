"""Aggregated configuration for executing a full training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config.constants import RESULTS_ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from .model import ModelOption
    from .training import TrainingOption

__all__ = ["ModelTrainingOption"]


@dataclass(slots=True)
class ModelTrainingOption:
    """
    Aggregated configuration used for end-to-end training experiments.

    - Couples a ``ModelOption`` with the dataloader-producing ``TrainingOption``.
    - Exposes filesystem helpers for recording params, metrics, splits, and
      checkpoint state dictionaries.
    - Construction time validation ensures the model’s ``target_kind`` matches
      the dataset so incompatibilities are caught long before GPU cycles are
      spent.
    """

    model_option: ModelOption
    training_option: TrainingOption

    def __post_init__(self) -> None:
        """Validate that the model and dataset target kinds agree."""
        data_option = self.training_option.training_data_option
        if self.model_option.target_kind != data_option.target_kind:
            raise ValueError(
                "model_option target_kind must match training_data_option target_kind.",
            )

    def get_path(self) -> Path:
        """Directory where model artifacts (weights + metrics) are written."""
        feature_option = (
            self.training_option.training_data_option.feature_extraction_option
        )
        feature_path = RESULTS_ROOT / feature_option.name
        return (
            feature_path
            / "models"
            / self.model_option.name
            / self.training_option.training_method_option.name
        )

    def get_params_path(self) -> Path:
        """Return the file used to persist ``to_params`` metadata."""
        return self.get_path() / "params.json"

    def get_metrics_path(self) -> Path:
        """Return the metrics JSON path."""
        return self.get_path() / "metrics.json"

    def get_state_dict_path(self) -> Path:
        """Return the path for the serialized model weights."""
        return self.get_path() / "best_model.pt"

    def get_splits_path(self) -> Path:
        """Return the path for persisting train/test segment identifiers."""
        return self.get_path() / "splits.json"

    def to_params(self) -> dict[str, Any]:
        """Serialize the aggregated model + training configuration."""
        return {
            "model_option": self.model_option.to_params(),
            "training_option": self.training_option.to_params(),
        }
