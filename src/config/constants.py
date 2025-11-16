from pathlib import Path

# Data layout
_source_paths: list[Path] = [
    DEAP_ROOT := Path("./data/DEAP"),
    DEAP_ORIGINAL := DEAP_ROOT / "DEAP-Dataset" / "data_original",
    DEAP_RATINGS_CSV := (
        DEAP_ROOT / "DEAP-Dataset" / "metadata_csv" / "participant_ratings.csv"
    ),
    DEAP_CHANNELS_XLSX := (
        DEAP_ROOT / "additional-information" / "DEAP_EEG_channels.xlsx"
    ),
]

for _path in _source_paths:
    if not _path.exists():
        raise FileNotFoundError(f"{_path} does not exist")

# DEAP specifics
TRIALS_NUM: int = 40
EEG_ELECTRODES_NUM: int = 32
SFREQ_TARGET: float = 128.0
EPOCH_TMIN: float = -5.0
EPOCH_TMAX: float = 60.0

# Feature extraction
SFREQ: int = 128
BASELINE_SEC: float = 5.0
TRIAL_SEC: float = 60.0
WINDOW_SEC: float = 2.0
STEP_SEC: float = 0.25

# Frequency bands (Hz)
BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "slow_alpha": (8.0, 10.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES: list[str] = list(BANDS.keys())

# Canonical Geneva 32-channel order
GENEVA_32: list[str] = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "Fz",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "Cz",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

# Left-right pairs for asymmetry
ASYM_PAIRS: list[tuple[str, str]] = [
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
    ("C3", "C4"),
    ("T7", "T8"),
    ("CP5", "CP6"),
    ("CP1", "CP2"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("PO3", "PO4"),
    ("O1", "O2"),
]
