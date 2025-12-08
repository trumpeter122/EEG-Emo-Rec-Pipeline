"""Shared helpers for handling regression vs. classification target kinds."""

from __future__ import annotations

from typing import Literal

import numpy as np

ClassificationKind = Literal["classification", "classification_5", "classification_3"]
TargetKind = Literal["regression", "classification", "classification_5", "classification_3"]

__all__ = [
    "ClassificationKind",
    "TargetKind",
    "classification_target_kinds",
    "collapse_classification_labels",
    "expected_class_labels",
    "is_classification_kind",
    "target_kinds_compatible",
]

classification_target_kinds: set[str] = {"classification", "classification_5", "classification_3"}


def is_classification_kind(target_kind: TargetKind) -> bool:
    """Return ``True`` when the target kind represents a classification task."""
    return target_kind in classification_target_kinds


def target_kinds_compatible(left: TargetKind, right: TargetKind) -> bool:
    """
    Determine whether two target kinds can operate together.

    - Any pair of classification variants are considered compatible so models
      and training methods declared for ``classification`` can be reused for
      ``classification_5``/``classification_3`` datasets.
    - Regression must match regression exactly.
    """
    if left == right:
        return True
    return left in classification_target_kinds and right in classification_target_kinds


def expected_class_labels(target_kind: TargetKind) -> list[float] | None:
    """Return the canonical label set for the provided target kind."""
    if not is_classification_kind(target_kind):
        return None
    if target_kind == "classification":
        return [float(index) for index in range(1, 10)]
    if target_kind == "classification_5":
        return [float(index) for index in range(1, 6)]
    if target_kind == "classification_3":
        return [float(index) for index in range(1, 4)]
    raise ValueError(f"Unsupported target_kind: {target_kind}")


def _collapse_to_five(values: np.ndarray) -> np.ndarray:
    """Map 9-point labels onto five bins: 1-2, 3-4, 5, 6-7, 8-9."""
    return np.select(
        [
            values <= 2.0,
            values <= 4.0,
            values <= 5.0,
            values <= 7.0,
            values > 7.0,
        ],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ).astype(np.float32)


def _collapse_to_three(values: np.ndarray) -> np.ndarray:
    """Map 9-point labels onto three bins: 1-3, 4-6, 7-9."""
    return np.select(
        [values <= 3.0, values <= 6.0, values > 6.0],
        [1.0, 2.0, 3.0],
    ).astype(np.float32)


def collapse_classification_labels(
    *,
    values: np.ndarray,
    target_kind: TargetKind,
) -> np.ndarray:
    """
    Collapse 9-class labels into the requested bins for derived classification tasks.

    Regression targets are passed through unchanged; standard classification
    keeps the original values, while ``classification_5`` and
    ``classification_3`` merge adjacent scores into five or three buckets.
    """
    if target_kind == "classification_5":
        return _collapse_to_five(values.astype(np.float32))
    if target_kind == "classification_3":
        return _collapse_to_three(values.astype(np.float32))
    return values
