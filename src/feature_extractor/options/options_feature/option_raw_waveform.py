"""Raw waveform segments for posterior visual encoding replication."""

from __future__ import annotations

import numpy as np

from config import GENEVA_32
from feature_extractor.types import FeatureOption

__all__ = ["_raw_waveform"]


def _extract_raw_waveform(
    trial_data: np.ndarray,
    channel_pick: list[str],
) -> np.ndarray:
    """
    Return zero-mean raw waveforms for the selected channels.
    """
    data = np.asarray(trial_data, dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = data[ch_indices, :]
    centered = picked - picked.mean(axis=1, keepdims=True)
    return centered.astype(np.float32)


_raw_waveform = FeatureOption(
    name="raw_waveform",
    feature_channel_extraction_method=_extract_raw_waveform,
)
