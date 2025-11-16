import numpy as np

from .option_de import _extract_de
from .utils import _available_pairs


def _extract_deasm(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """
    Compute DEASM (differential entropy asymmetry) for one segment.
    """
    de_feats = _extract_de(trial_data, channel_pick)  # (n_channels_used, n_bands)
    pairs = _available_pairs(channel_pick)
    out = np.zeros((len(pairs), de_feats.shape[1]), dtype=np.float32)
    for pi, (li, ri, _) in enumerate(pairs):
        out[pi] = de_feats[li] - de_feats[ri]
    return out
