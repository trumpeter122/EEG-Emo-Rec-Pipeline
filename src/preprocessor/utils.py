from pathlib import Path

import mne

from config import (
    DEAP_ORIGINAL,
)


# -----------------------------
# File Operation Utils
# -----------------------------
def _bdf_path(subject_id: int) -> Path:
    return DEAP_ORIGINAL / f"s{subject_id:02d}.bdf"


def _subject_npy_path(folder: Path, subject_id: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"s{subject_id:02}.npy"


def _load_raw_subject(subject_id: int) -> mne.io.BaseRaw:
    return mne.io.read_raw_bdf(
        _bdf_path(subject_id),
        preload=True,
        verbose=False,
    ).load_data()
