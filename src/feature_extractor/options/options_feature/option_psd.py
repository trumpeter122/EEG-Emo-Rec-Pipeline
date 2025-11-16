import numpy as np
from scipy import signal

from config import (
    BANDS,
    GENEVA_32,
    SFREQ,
)


def _extract_psd(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """
    Compute band-power (log-PSD) features for one segment.

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

    n_ch, n_samples = x.shape
    n_bands = len(BANDS)
    feats = np.zeros((n_ch, n_bands), dtype=np.float32)

    for ci in range(n_ch):
        # Welch PSD in µV^2, log10
        nperseg = min(SFREQ, n_samples)
        freqs, Pxx = signal.welch(x=x[ci], fs=SFREQ, nperseg=nperseg)
        Pxx = Pxx * (1e6**2)
        Pxx = 10.0 * np.log10(Pxx + 1e-12)

        # Band averages (same as _band_means in __init__.py)
        band_vals = []
        for fmin, fmax in BANDS.values():
            mask = (freqs >= fmin) & (freqs < fmax)
            if np.any(mask):
                band_vals.append(float(np.mean(Pxx[mask])))
            else:
                band_vals.append(np.nan)
        feats[ci, :] = np.asarray(band_vals, dtype=np.float32)

    return feats
