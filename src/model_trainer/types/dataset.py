"""Dataset abstraction for segment-level feature tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["SegmentDataset"]


@dataclass(slots=True)
class SegmentDataset(Dataset[tuple[np.ndarray, float]]):
    """
    ``torch.utils.data.Dataset`` wrapper for normalized features and targets.

    - Segment-level feature arrays are converted to contiguous ``float32``
      vectors so CNN models can later reshape them consistently.
    - Targets are stored using the dtype requested by the owning
      ``TrainingDataOption`` so both regression (float) and classification (int)
      scenarios are supported.
    - The initializer validates feature/target lengths to catch preprocessing
      mistakes before PyTorch ever sees the data.
    """

    _features: list[np.ndarray]
    _targets: np.ndarray

    def __init__(
        self,
        *,
        features: Sequence[np.ndarray],
        targets: Sequence[float],
        target_dtype: np.dtype[Any],
    ):
        """
        Initialize the SegmentDataset.

        Args:
        ----
            features: A sequence of feature arrays.
            targets: A sequence of target values.
            target_dtype: The desired data type for the targets.
        """
        if len(features) != len(targets):
            raise ValueError(
                "features and targets must contain the same number of rows.",
            )

        self._features = [
            np.asarray(feature, dtype=np.float32).reshape(-1) for feature in features
        ]
        self._targets = np.asarray(list(targets), dtype=target_dtype)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
        -------
            The number of samples.
        """
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float]:
        """
        Return the feature and target for a given index.

        Args:
        ----
            index: The index of the item to retrieve.

        Returns:
        -------
            A tuple containing the feature array and the target value.
        """
        return self._features[index], self._targets[index]
