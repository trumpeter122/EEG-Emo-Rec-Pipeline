"""Feature extraction option definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config.constants import BASELINE_SEC, GENEVA_32
from config.option_utils import FeatureChannelExtractionMethod, _callable_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from preprocessor.types import PreprocessingOption

__all__ = [
    "ChannelPickOption",
    "FeatureExtractionOption",
    "FeatureOption",
    "SegmentationOption",
]


@dataclass(slots=True)
class ChannelPickOption:
    """Subset of EEG channels retained for feature extraction."""

    name: str
    channel_pick: list[str]

    def __post_init__(self) -> None:
        invalid = [
            channel_name
            for channel_name in self.channel_pick
            if channel_name not in GENEVA_32
        ]
        if invalid:
            raise ValueError(f"Invalid channel names: {invalid}")

    def to_params(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the option."""
        return {
            "name": self.name,
            "channel_pick": list(self.channel_pick),
        }


@dataclass(slots=True)
class FeatureOption:
    """Feature extraction method wrapper."""

    name: str
    feature_channel_extraction_method: FeatureChannelExtractionMethod

    def to_params(self) -> dict[str, Any]:
        """Serialize the feature option metadata."""
        return {
            "name": self.name,
            "feature_channel_extraction_method": _callable_path(
                self.feature_channel_extraction_method,
            ),
        }


@dataclass(slots=True)
class SegmentationOption:
    """Sliding-window configuration for slicing each trial."""

    time_window: float
    time_step: float
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.time_window <= 0:
            raise ValueError("time_window must be positive.")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive.")
        if self.time_step > self.time_window:
            raise ValueError("time_step cannot exceed time_window.")
        if self.time_window > BASELINE_SEC:
            raise ValueError("time_window cannot exceed the baseline duration.")

        self.name = f"w{self.time_window:.2f}_s{self.time_step:.2f}"

    def to_params(self) -> dict[str, Any]:
        """Serialize the segmentation configuration."""
        return {
            "name": self.name,
            "time_window": self.time_window,
            "time_step": self.time_step,
        }


@dataclass(slots=True)
class FeatureExtractionOption:
    """Fully-qualified feature extraction pipeline configuration."""

    preprocessing_option: PreprocessingOption
    feature_option: FeatureOption
    channel_pick_option: ChannelPickOption
    segmentation_option: SegmentationOption
    name: str = field(init=False)
    extraction_method: Callable[[np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.name = "+".join(
            [
                self.preprocessing_option.name,
                self.feature_option.name,
                self.channel_pick_option.name,
                self.segmentation_option.name,
            ],
        )
        self.extraction_method = self._build_extraction_method()

    def _build_extraction_method(self) -> Callable[[np.ndarray], np.ndarray]:
        """Partially apply the feature extractor with the configured channels."""

        def _extract(trial_data: np.ndarray) -> np.ndarray:
            return self.feature_option.feature_channel_extraction_method(
                trial_data=trial_data,
                channel_pick=self.channel_pick_option.channel_pick,
            )

        return _extract

    def get_path(self) -> Path:
        """Return the directory storing extracted segment features."""
        path = self.preprocessing_option.get_feature_path() / self.name
        path.mkdir(exist_ok=True)
        return path

    def get_file_paths(self) -> list[Path]:
        """Return sorted joblib file paths with extracted segment features."""
        paths = list(self.get_path().glob("*.joblib"))
        if not paths:
            raise FileNotFoundError(
                f'No feature data found at "{self.get_path()}" for option {self.name}.',
            )
        return sorted(paths)

    def get_metadata_path(self) -> Path:
        """Return the path to the metadata directory."""
        return self.get_path() / "metadata"

    def get_metadata_baseline_path(self) -> Path:
        """Return the path to the baseline metadata directory."""
        return self.get_metadata_path() / "baseline"

    def get_metadata_params_path(self) -> Path:
        """Return the path to the parameters JSON file."""
        return self.get_metadata_path() / "params.json"

    def get_metadata_metrics_path(self) -> Path:
        """Return the path to the metrics JSON file."""
        return self.get_metadata_path() / "metrics.json"

    def get_metadata_shape_path(self) -> Path:
        """Return the path to the shape CSV file."""
        return self.get_metadata_path() / "shape.csv"

    def to_params(self) -> dict[str, Any]:
        """Serialize the aggregation of the underlying option metadata."""
        return {
            "name": self.name,
            "preprocessing_option": self.preprocessing_option.to_params(),
            "feature_option": self.feature_option.to_params(),
            "channel_pick_option": self.channel_pick_option.to_params(),
            "segmentation_option": self.segmentation_option.to_params(),
        }
