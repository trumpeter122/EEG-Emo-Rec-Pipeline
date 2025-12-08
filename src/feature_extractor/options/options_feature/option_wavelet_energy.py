"""Wavelet energy, entropy, and statistical features (db4, 5 levels)."""

from __future__ import annotations

import numpy as np

from config import GENEVA_32
from feature_extractor.types import FeatureOption

__all__ = ["_wavelet_energy_entropy_stats"]

_EPS = 1e-12

# Daubechies-4 (db4) decomposition filters.
_DB4_DEC_LO = np.array(
    [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171909308,
        -0.02798376941698385,
        0.6308807679298587,
        0.7148465705529154,
        0.23037781330885523,
    ],
    dtype=np.float64,
)
_DB4_DEC_HI = np.array(
    [
        -0.23037781330885523,
        0.7148465705529154,
        -0.6308807679298587,
        -0.02798376941698385,
        0.18703481171909308,
        0.030841381835986965,
        -0.032883011666982945,
        -0.010597401784997278,
    ],
    dtype=np.float64,
)

_BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def _dwt_single_level(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform one level of db4 DWT using symmetric padding."""
    pad_len = _DB4_DEC_LO.size - 1
    left = pad_len // 2
    right = pad_len - left
    padded = np.pad(signal, (left, right), mode="symmetric")
    approx = np.convolve(padded, _DB4_DEC_LO[::-1], mode="valid")[::2]
    detail = np.convolve(padded, _DB4_DEC_HI[::-1], mode="valid")[::2]
    return approx, detail


def _band_coefficients(signal: np.ndarray) -> dict[str, np.ndarray]:
    """Return db4 coefficients mapped to EEG bands."""
    approx = signal.astype(np.float64)
    details: list[np.ndarray] = []
    for _ in range(5):
        approx, detail = _dwt_single_level(approx)
        details.append(detail)

    return {
        "delta": approx,  # A5 -> ~0-4 Hz
        "theta": details[4],  # D5 -> ~4-8 Hz
        "alpha": details[3],  # D4 -> ~8-16 Hz
        "beta": details[2],  # D3 -> ~16-32 Hz
        "gamma": details[1],  # D2 -> ~32-64 Hz
    }


def _band_energy(coeffs: np.ndarray) -> float:
    """Sum of squared coefficients."""
    return float(np.sum(coeffs * coeffs))


def _band_entropy(coeffs: np.ndarray) -> float:
    """Shannon entropy of normalized coefficient energies."""
    power = coeffs * coeffs
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probs = power / (total + _EPS)
    return float(-np.sum(probs * np.log(probs + _EPS)))


def _channel_features(channel_data: np.ndarray) -> np.ndarray:
    """Compute wavelet-based energies/ratios and statistical moments."""
    channel_data = np.asarray(channel_data, dtype=np.float64)
    coeffs = _band_coefficients(signal=channel_data)

    energies = {band: _band_energy(coeff) for band, coeff in coeffs.items()}
    total_energy = float(sum(energies.values())) + _EPS

    features: list[float] = []
    for band in _BAND_ORDER:
        energy = energies[band]
        features.append(energy)
    for band in _BAND_ORDER:
        energy = energies[band]
        ratio = energy / total_energy
        features.append(ratio)
    for band in _BAND_ORDER:
        energy = energies[band]
        ratio = energy / total_energy
        features.append(float(np.log10(ratio + _EPS)))
    for band in _BAND_ORDER:
        energy = energies[band]
        ratio = energy / total_energy
        features.append(float(np.abs(np.log10(ratio + _EPS))))
    for band in _BAND_ORDER:
        features.append(_band_entropy(coeffs[band]))

    mean = float(np.mean(channel_data))
    std = float(np.std(channel_data))
    denom = std if std > 0.0 else 1.0

    first_diff = np.diff(channel_data)
    second_diff = np.diff(channel_data, n=2)

    features.extend(
        [
            mean,
            std,
            float(np.mean(np.abs(first_diff))),
            float(np.mean(np.abs(first_diff)) / (denom + _EPS)),
            float(np.mean(np.abs(second_diff))),
            float(np.mean(np.abs(second_diff)) / (denom + _EPS)),
        ]
    )

    return np.asarray(features, dtype=np.float32)


def _extract_wavelet_features(
    trial_data: np.ndarray,
    channel_pick: list[str],
) -> np.ndarray:
    """Extract per-channel wavelet energy/entropy/statistical features."""
    data = np.asarray(trial_data)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = data[ch_indices, :]
    channel_features = [
        _channel_features(channel_data=picked[ch_idx])
        for ch_idx in range(picked.shape[0])
    ]
    return np.stack(channel_features, axis=0)


_wavelet_energy_entropy_stats = FeatureOption(
    name="wavelet_energy_entropy_stats",
    feature_channel_extraction_method=_extract_wavelet_features,
)
