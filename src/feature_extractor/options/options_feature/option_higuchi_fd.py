"""Higuchi fractal-dimension features for low-channel frontal montage."""

from __future__ import annotations

import numpy as np

from config import GENEVA_32
from feature_extractor.types import FeatureOption

__all__ = ["_higuchi_fd_frontal"]


def _higuchi_fd(signal: np.ndarray, k_max: int) -> float:
    """
    Compute Higuchi's fractal dimension for a 1D signal.

    The implementation follows the original definition with evenly spaced
    subsamples and log-log regression of curve length vs. k.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < k_max + 2:
        raise ValueError("Signal too short for requested k_max.")

    l_k: list[float] = []
    for k in range(1, k_max + 1):
        lk_sum = 0.0
        for m in range(k):
            idx = slice(m, n, k)
            x_mk = x[idx]
            if x_mk.size < 2:
                continue
            diff = np.abs(np.diff(x_mk)).sum()
            norm = (n - 1) / (x_mk.size * k)
            lk_sum += norm * diff
        lk_avg = lk_sum / k
        l_k.append(lk_avg)

    ks = np.log(np.arange(1, k_max + 1, dtype=np.float64))
    lks = np.log(np.asarray(l_k) + 1e-12)
    coeffs = np.polyfit(ks, lks, deg=1)
    return float(coeffs[0])


def _extract_higuchi_fd(
    trial_data: np.ndarray,
    channel_pick: list[str],
) -> np.ndarray:
    """Return per-channel fractal dimension + asymmetry (AF3-F4)."""
    data = np.asarray(trial_data)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = data[ch_indices, :]

    fd_values = [
        _higuchi_fd(signal=picked[ch_idx, :], k_max=8)
        for ch_idx in range(picked.shape[0])
    ]
    features = np.asarray(fd_values, dtype=np.float32)[:, None]

    if "AF3" in channel_pick and "F4" in channel_pick:
        left_fd = fd_values[channel_pick.index("AF3")]
        right_fd = fd_values[channel_pick.index("F4")]
        asym = np.asarray([[left_fd - right_fd]], dtype=np.float32)
        features = np.vstack([features, asym])

    return features


_higuchi_fd_frontal = FeatureOption(
    name="higuchi_fd_frontal",
    feature_channel_extraction_method=_extract_higuchi_fd,
)
