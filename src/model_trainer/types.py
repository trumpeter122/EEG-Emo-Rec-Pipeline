"""Training and model option definitions for model_trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    MinMaxScaler,
    StandardScaler,
)
from torch.utils.data import DataLoader, Dataset

from config.constants import RESULTS_ROOT
from config.option_utils import (
    CriterionBuilder,
    ModelBuilder,
    OptimizerBuilder,
    _callable_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from feature_extractor.types import FeatureExtractionOption

__all__ = [
    "ModelOption",
    "ModelTrainingOption",
    "TrainingDataOption",
    "TrainingMethodOption",
    "TrainingOption",
]

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer


class _SegmentDataset(Dataset[tuple[np.ndarray, float]]):
    """Minimal torch ``Dataset`` wrapper for segment features and targets."""

    def __init__(
        self,
        *,
        features: Sequence[np.ndarray],
        targets: Sequence[float],
        target_dtype: np.dtype[Any],
    ):
        if len(features) != len(targets):
            raise ValueError(
                "features and targets must contain the same number of rows.",
            )

        self._features = [
            np.asarray(feature, dtype=np.float32).reshape(-1) for feature in features
        ]
        self._targets = np.asarray(list(targets), dtype=target_dtype)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float]:
        return self._features[index], self._targets[index]


@dataclass(slots=True)
class TrainingDataOption:
    """Configuration describing how extracted features are transformed into datasets."""

    feature_extraction_option: FeatureExtractionOption
    target: str
    random_seed: int
    use_size: float
    test_size: float
    target_kind: Literal["regression", "classification"]
    feature_scaler: Literal["none", "standard", "minmax"]
    class_labels_expected: Sequence[float] | None = None
    name: str = field(init=False)
    train_dataset: Dataset[tuple[np.ndarray, float]] = field(init=False)
    test_dataset: Dataset[tuple[np.ndarray, float]] = field(init=False)
    segment_splits: dict[str, list[int]] = field(init=False)
    class_labels: list[float] | None = field(init=False, default=None)
    _target_dtype: np.dtype[Any] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_sizes()
        self.name = self._build_name()

        frame = self._load_feature_frame()
        trimmed = cast(
            "pd.DataFrame",
            frame.loc[:, ["data", self.target]].reset_index(drop=True),
        )
        trimmed = self._encode_targets(frame=trimmed)
        trimmed = self._scale_feature_column(frame=trimmed)

        splits = self._generate_segment_splits(total=len(trimmed))
        self.segment_splits = splits
        self.train_dataset = self._build_dataset(
            frame=trimmed,
            indices=splits["train-segments"],
        )
        self.test_dataset = self._build_dataset(
            frame=trimmed,
            indices=splits["test-segments"],
        )

    def _validate_sizes(self) -> None:
        if not 0 < self.use_size <= 1:
            raise ValueError("use_size must fall within the interval (0, 1].")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must fall within the interval (0, 1).")

    def _build_name(self) -> str:
        return (
            f"{self.feature_extraction_option.name}"
            f"|target={self.target}"
            f"|use={self.use_size:.2f}"
            f"|test={self.test_size:.2f}"
            f"|seed={self.random_seed}"
        )

    def _load_feature_frame(self) -> pd.DataFrame:
        file_paths = self.feature_extraction_option.get_file_paths()
        frames: list[pd.DataFrame] = []
        for filename in file_paths:
            frame_obj = joblib.load(filename=filename)
            if not isinstance(frame_obj, pd.DataFrame):
                raise TypeError(
                    f"Feature file {filename} did not contain a pandas DataFrame.",
                )
            frames.append(frame_obj)

        if not frames:
            raise ValueError(
                "No feature data found for option "
                f"{self.feature_extraction_option.name}.",
            )

        frame = cast("pd.DataFrame", pd.concat(frames, axis=0, ignore_index=True))
        if self.target not in frame.columns:
            raise KeyError(
                f'Target column "{self.target}" is missing from the features frame.',
            )
        if "data" not in frame.columns:
            raise KeyError('Column "data" is missing from the features frame.')

        return frame

    def _encode_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.loc[:, self.target].to_numpy(copy=True)
        if self.target_kind == "classification":
            if self.class_labels_expected is not None:
                unique_values = [float(value) for value in self.class_labels_expected]
            else:
                unique_values = sorted({float(value) for value in values})
            if len(unique_values) < 2:
                raise ValueError(
                    "classification targets must contain at least two classes.",
                )
            if self.class_labels_expected is not None:
                label_array = np.asarray(unique_values, dtype=np.float32)
                value_array = values.astype(np.float32)
                encoded = (
                    np.abs(value_array[:, None] - label_array[None, :])
                    .argmin(
                        axis=1,
                    )
                    .astype(np.int64)
                )
            else:
                mapping = {label: index for index, label in enumerate(unique_values)}
                encoded = np.asarray(
                    [mapping[float(value)] for value in values],
                    dtype=np.int64,
                )
            self.class_labels = unique_values
            self._target_dtype = np.dtype(np.int64)
            frame = frame.copy()
            frame.loc[:, self.target] = encoded
        elif self.target_kind == "regression":
            frame = frame.copy()
            frame.loc[:, self.target] = values.astype(np.float32)
            self.class_labels = None
            self._target_dtype = np.dtype(np.float32)
        else:
            raise ValueError(f"Unsupported target_kind: {self.target_kind}")
        return frame

    def _scale_feature_column(self, frame: pd.DataFrame) -> pd.DataFrame:
        arrays = [
            np.asarray(feature, dtype=np.float32) for feature in frame["data"].tolist()
        ]
        if not arrays:
            raise ValueError("No feature arrays were found to scale.")

        if self.feature_scaler == "none":
            frame = frame.copy()
            frame.loc[:, "data"] = arrays
            return frame

        flattened = np.stack([array.reshape(-1) for array in arrays], axis=0)
        if self.feature_scaler == "standard":
            scaler = StandardScaler()
        elif self.feature_scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported feature_scaler: {self.feature_scaler}")
        scaled = scaler.fit_transform(flattened)
        reshaped = [
            scaled[row_index].reshape(arrays[row_index].shape)
            for row_index in range(len(arrays))
        ]
        frame = frame.copy()
        frame.loc[:, "data"] = reshaped
        return frame

    def _generate_segment_splits(self, total: int) -> dict[str, list[int]]:
        if total < 2:
            raise ValueError("At least two segments are required for training.")

        usable = max(1, int(total * self.use_size))
        if usable < 2:
            raise ValueError(
                "use_size selects fewer than two segments; increase use_size.",
            )

        rng = random.Random(self.random_seed)
        indices = list(range(total))
        rng.shuffle(indices)
        used_indices = indices[:usable]

        test_count = max(1, int(round(len(used_indices) * self.test_size)))
        if test_count >= len(used_indices):
            raise ValueError(
                "test_size allocates all usable segments to testing; "
                "decrease test_size or increase use_size.",
            )

        test_indices = sorted(used_indices[:test_count])
        train_indices = sorted(used_indices[test_count:])

        return {
            "train-segments": train_indices,
            "test-segments": test_indices,
        }

    def _build_dataset(
        self,
        *,
        frame: pd.DataFrame,
        indices: list[int],
    ) -> Dataset[tuple[np.ndarray, float]]:
        subset = frame.iloc[indices].reset_index(drop=True)
        features = subset["data"].tolist()
        targets = subset[self.target].tolist()
        return _SegmentDataset(
            features=features,
            targets=targets,
            target_dtype=self._target_dtype,
        )

    def to_params(self) -> dict[str, Any]:
        """Serialize training configuration metadata."""
        return {
            "name": self.name,
            "feature_extraction_option": self.feature_extraction_option.to_params(),
            "target": self.target,
            "random_seed": self.random_seed,
            "use_size": self.use_size,
            "test_size": self.test_size,
            "target_kind": self.target_kind,
            "feature_scaler": self.feature_scaler,
            "class_labels": list(self.class_labels) if self.class_labels else None,
            "class_labels_expected": (
                list(self.class_labels_expected)
                if self.class_labels_expected is not None
                else None
            ),
        }

    def get_class_values(self) -> np.ndarray | None:
        if not self.class_labels:
            return None
        return np.asarray(self.class_labels, dtype=np.float32)


@dataclass(slots=True)
class TrainingMethodOption:
    """Optimization hyperparameters controlling how datasets are consumed."""

    name: str
    epochs: int
    batch_size: int
    optimizer_builder: OptimizerBuilder
    criterion_builder: CriterionBuilder
    num_workers: int
    pin_memory: bool
    drop_last: bool
    target_kind: Literal["regression", "classification"]

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

    def build_optimizer(self, *, model: nn.Module) -> Optimizer:
        """Instantiate the optimizer for ``model``."""
        return self.optimizer_builder(model=model)

    def build_criterion(self) -> nn.Module:
        """Instantiate the configured loss function."""
        return self.criterion_builder()

    def build_dataloader(
        self,
        *,
        dataset: Dataset[tuple[np.ndarray, float]],
        shuffle: bool,
    ) -> DataLoader[tuple[np.ndarray, float]]:
        """Create a ``DataLoader`` suitable for the configured training regime."""
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
        }


@dataclass(slots=True)
class TrainingOption:
    """Consolidated configuration describing datasets + optimization strategy."""

    training_data_option: TrainingDataOption
    training_method_option: TrainingMethodOption
    name: str = field(init=False)
    train_loader: DataLoader[tuple[np.ndarray, float]] = field(init=False)
    test_loader: DataLoader[tuple[np.ndarray, float]] = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.training_method_option.target_kind
            != self.training_data_option.target_kind
        ):
            raise ValueError(
                "training_method_option target_kind must match training_data_option.",
            )
        self.name = (
            f"{self.training_data_option.name}"
            f"|method={self.training_method_option.name}"
        )
        self.train_loader = self.training_method_option.build_dataloader(
            dataset=self.training_data_option.train_dataset,
            shuffle=True,
        )
        self.test_loader = self.training_method_option.build_dataloader(
            dataset=self.training_data_option.test_dataset,
            shuffle=False,
        )

    def to_params(self) -> dict[str, Any]:
        """Serialize the combined training configuration."""
        return {
            "name": self.name,
            "training_data_option": self.training_data_option.to_params(),
            "training_method_option": self.training_method_option.to_params(),
        }


@dataclass(slots=True)
class ModelOption:
    """Abstraction describing how to construct a specific neural network model."""

    name: str
    model_builder: ModelBuilder
    output_size: int
    target_kind: Literal["regression", "classification"]
    model_args: tuple[Any, ...] = field(default_factory=tuple)
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_model(self) -> nn.Module:
        """Instantiate the configured model."""
        kwargs = dict(self.model_kwargs)
        kwargs.setdefault("output_size", self.output_size)
        return self.model_builder(*self.model_args, **kwargs)

    def to_params(self) -> dict[str, Any]:
        """Serialize the model descriptor."""
        return {
            "name": self.name,
            "model_builder": _callable_path(self.model_builder),
            "output_size": self.output_size,
            "target_kind": self.target_kind,
            "model_args": list(self.model_args),
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(slots=True)
class ModelTrainingOption:
    """Aggregated configuration used for end-to-end training experiments."""

    model_option: ModelOption
    training_option: TrainingOption

    def __post_init__(self) -> None:
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
