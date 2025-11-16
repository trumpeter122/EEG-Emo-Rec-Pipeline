import numpy as np

from .option_psd import _extract_psd
from .utils import _available_pairs


def _extract_dasm(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """
    Compute DASM (differential asymmetry) on top of PSD features for one segment.
    """
    psd_feats = _extract_psd(trial_data, channel_pick)  # (n_channels_used, n_bands)
    pairs = _available_pairs(channel_pick)
    out = np.zeros((len(pairs), psd_feats.shape[1]), dtype=np.float32)
    for pi, (li, ri, _) in enumerate(pairs):
        out[pi] = psd_feats[li] - psd_feats[ri]
    return out
