"""Higher-order crossings + statistical + Higuchi FD features."""

from __future__ import annotations

import numpy as np

from config import GENEVA_32
from feature_extractor.types import FeatureOption

__all__ = ["_hoc_stat_fd_frontal4"]

_EPS = 1e-12


def _higuchi_fd(signal: np.ndarray, k_max: int) -> float:
    """Compute Higuchi's fractal dimension for a 1D signal."""
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < k_max + 2:
        raise ValueError("Signal too short for Higuchi FD.")

    lengths: list[float] = []
    for k in range(1, k_max + 1):
        lk_sum = 0.0
        for m in range(k):
            subseq = x[m::k]
            if subseq.size < 2:
                continue
            diff = np.abs(np.diff(subseq)).sum()
            norm = (n - 1) / (subseq.size * k)
            lk_sum += norm * diff
        lk = lk_sum / k if k > 0 else 0.0
        lengths.append(lk)

    ks = np.log(np.arange(1, k_max + 1, dtype=np.float64))
    lks = np.log(np.asarray(lengths, dtype=np.float64) + _EPS)
    slope, _intercept = np.polyfit(ks, lks, deg=1)
    return float(-slope)


def _hoc_features(signal: np.ndarray, k_max: int) -> np.ndarray:
    """Compute higher-order crossings D1..Dk for a zero-mean signal."""
    x = signal - np.mean(signal)
    features: list[float] = []
    for order in range(1, k_max + 1):
        diff = x if order == 1 else np.diff(x, n=order - 1)
        if diff.size < 2:
            features.append(0.0)
            continue
        sign = diff >= 0.0
        crossings = np.count_nonzero(sign[1:] != sign[:-1])
        features.append(float(crossings))
    return np.asarray(features, dtype=np.float32)


def _stat_features(signal: np.ndarray) -> np.ndarray:
    """Six statistical features as defined in the paper."""
    x = np.asarray(signal, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x)) + _EPS
    first_diff = np.diff(x)
    second_diff = np.diff(x, n=2)
    delta = float(np.mean(np.abs(first_diff)))
    gamma = float(np.mean(np.abs(second_diff))) if second_diff.size else 0.0
    return np.asarray(
        [
            mean,
            std,
            delta,
            delta / std,
            gamma,
            gamma / std,
        ],
        dtype=np.float32,
    )


def _extract_hoc_stat_fd(
    trial_data: np.ndarray,
    channel_pick: list[str],
) -> np.ndarray:
    """
    Extract HOC+statistical+Higuchi FD features per channel.

    - Channels: FC5, F4, F7, AF3 (paper's 4-electrode set).
    - Window: handled by segmentation; this method works on one segment.
    - HOC order: k_max=36.
    - Normalization: z-score per feature across channels.
    """
    data = np.asarray(trial_data)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = data[ch_indices, :]

    features: list[np.ndarray] = []
    for channel_idx in range(picked.shape[0]):
        channel = picked[channel_idx, :]
        stat = _stat_features(channel)
        hoc = _hoc_features(channel, k_max=36)
        fd = np.asarray([_higuchi_fd(channel, k_max=8)], dtype=np.float32)
        channel_features = np.concatenate([hoc, stat, fd], dtype=np.float32)
        features.append(channel_features)

    stacked = np.vstack(features)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + _EPS
    normalized = (stacked - mean) / std
    return normalized.astype(np.float32)


_hoc_stat_fd_frontal4 = FeatureOption(
    name="hoc_stat_fd_frontal4",
    feature_channel_extraction_method=_extract_hoc_stat_fd,
)
