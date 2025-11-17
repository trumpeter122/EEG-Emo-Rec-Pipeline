import math

import numpy as np
from scipy import signal

from config import (
    GENEVA_32,
    SFREQ,
)

# Match the reference DEAP pipeline bandpass configuration.
DE_BANDS: dict[str, tuple[int, int]] = {
    "theta": (4, 8),
    "slow_alpha": (8, 10),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}


def _extract_de(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """
    Compute differential entropy (DE) features for one segment.

    Parameters
    ----------
    trial_data : np.ndarray
        Array of shape (n_channels_total, n_samples) for a single time segment.
    channel_pick : List[str]
        Channel names to use (subset of GENEVA_32).

    Returns
    -------
    np.ndarray
        Features of shape (n_channels_used, n_bands).
    """
    # Ensure 2D
    trial_data = np.asarray(trial_data)
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    # Pick channels by name
    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    x = trial_data[ch_indices, :]  # (n_channels_used, n_samples)

    n_ch, _ = x.shape
    n_bands = len(DE_BANDS)
    feats = np.zeros((n_ch, n_bands), dtype=np.float32)

    nyq = 0.5 * SFREQ

    for bi, (_, (fmin, fmax)) in enumerate(DE_BANDS.items()):
        low, high = fmin / nyq, fmax / nyq
        b, a = signal.butter(N=4, Wn=[low, high], btype="band")

        # Bandpass filter for all channels in this band
        band_sig = signal.filtfilt(b=b, a=a, x=x, axis=1)
        band_sig *= 1e6  # convert to microvolts to match reference scaling

        # DE = 0.5 * log(2πeσ²), same as compute_de_features in __init__.py
        stds = np.std(band_sig, axis=1) + 1e-12
        de = 0.5 * np.log10(2 * math.pi * math.e * (stds**2))
        feats[:, bi] = de.astype(np.float32)

    return feats
