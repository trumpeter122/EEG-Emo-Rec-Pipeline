import joblib
import numpy as np
import pandas as pd

from config import DEAP_RATINGS_CSV, TRIALS_NUM, PreprocessingOption
from utils import track

from .utils import (
    _load_raw_subject,
    _subject_npy_path,
)


# -----------------------------
# Batch Preprocessing Util
# -----------------------------
def _preprocess_subjects(
    preprocessing_option: PreprocessingOption,
) -> None:
    out_folder = preprocessing_option.get_subject_path()

    for subject_id in track(
        iterable=range(1, 33),
        description=f"Preprocessing with option `{preprocessing_option.name}`",
        context="Preprocessor",
    ):
        out_path = _subject_npy_path(folder=out_folder, subject_id=subject_id)
        if out_path.exists():
            continue

        # R: load raw and basic preparation
        raw = _load_raw_subject(subject_id=subject_id)

        # processing-only function
        data_out = preprocessing_option.preprocessing_method(
            raw,
            subject_id,
        )

        # W: save result in standard layout
        np.save(out_path, data_out)


# -----------------------------
# Trial Splitting and Metadata Exporting Util
# -----------------------------
def _split_trials(preprocessing_option: PreprocessingOption) -> None:
    source_folder = preprocessing_option.get_subject_path()
    target_folder = preprocessing_option.get_trial_path()
    ratings_csv_path = DEAP_RATINGS_CSV

    npy_files = sorted(f for f in source_folder.iterdir() if f.suffix == ".npy")
    ratings = pd.read_csv(ratings_csv_path)

    trial_counter = 0

    for f in track(
        iterable=npy_files,
        description="Splitting subject into trials for "
        f"option `{preprocessing_option.name}`",
        context="Preprocessor",
    ):
        subject_id = int(f.stem[1:3])
        data = np.load(f)  # (40, 32, T)
        subj_ratings = ratings[ratings["Participant_id"] == subject_id].sort_values(
            by="Experiment_id",
        )

        for i in range(TRIALS_NUM):
            trial_data = np.squeeze(data[i])

            row = subj_ratings.iloc[i]

            trial_df = pd.DataFrame(
                [
                    {
                        "data": trial_data,
                        "subject": int(row["Participant_id"]),
                        "trial": int(row["Trial"]),
                        "experiment_id": int(row["Experiment_id"]),
                        "valence": float(row["Valence"]),
                        "arousal": float(row["Arousal"]),
                        "dominance": float(row["Dominance"]),
                        "liking": float(row["Liking"]),
                    },
                ],
            )

            trial_counter += 1
            out_name = f"t{trial_counter:04}.joblib"
            out_path = target_folder / out_name
            if not out_path.exists():
                joblib.dump(trial_df, out_path, compress=3)


def run_preprocessor(preprocessing_option):
    _preprocess_subjects(preprocessing_option=preprocessing_option)

    _split_trials(preprocessing_option)
