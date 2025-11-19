"""Typed configuration objects for preprocessing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from config.constants import DEAP_ROOT

if TYPE_CHECKING:
    from config.option_utils import PreprocessingMethod

__all__ = ["PreprocessingOption"]


@dataclass(slots=True)
class PreprocessingOption:
    """Container describing a specific preprocessing configuration."""

    name: str
    root_dir: str | Path
    preprocessing_method: PreprocessingMethod
    root_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root_dir_path = Path(self.root_dir)
        self.root_path = DEAP_ROOT / "generated" / root_dir_path
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_subject_path(self) -> Path:
        """Directory for subject-level numpy arrays."""
        path = self.root_path / "subject"
        path.mkdir(exist_ok=True)
        return path

    def get_trial_path(self) -> Path:
        """Directory for per-trial joblib data frames produced by splitting."""
        path = self.root_path / "trial"
        path.mkdir(exist_ok=True)
        return path

    def get_feature_path(self) -> Path:
        """Directory for storing extracted feature files (joblib + baseline)."""
        path = self.root_path / "feature"
        path.mkdir(exist_ok=True)
        return path

    def to_params(self) -> dict[str, dict[str, str]]:
        """Serialize the option metadata into a JSON-friendly dictionary."""
        return {
            "preprocessing option": {
                "name": self.name,
                "root_path": str(self.root_path),
                "subject_path": str(self.get_subject_path()),
                "trial_path": str(self.get_trial_path()),
                "feature_path": str(self.get_feature_path()),
            },
        }
