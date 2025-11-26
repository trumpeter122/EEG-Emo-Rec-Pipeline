"""Preprocessing entry points for the DEAP pipeline."""

from __future__ import annotations

import csv
import json
import time
from typing import TYPE_CHECKING, Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from config import DEAP_RATINGS_CSV, TRIALS_NUM
from utils import message, track
from utils.fs import atomic_directory, directory_is_populated

if TYPE_CHECKING:
    from preprocessor.types import PreprocessingOption

from .utils import (
    _load_raw_subject,
    _subject_npy_path,
)

__all__ = ["run_preprocessor"]


def _shape_to_str(shape: tuple[int, ...]) -> str:
    """Represent a numpy-like shape tuple as e.g. ``32x8064``."""
    return "x".join(str(dim) for dim in shape) if shape else "()"


def _preprocess_subjects(
    preprocessing_option: PreprocessingOption,
) -> tuple[list[dict[str, Any]], float]:
    """
    Convert raw BDF files into subject-level numpy arrays atomically.

    Each subject is processed only once; if the destination file already exists
    the subject is skipped to avoid redundant computation.
    """
    out_folder = preprocessing_option.get_subject_path()
    if directory_is_populated(path=out_folder):
        message(description=f'"{out_folder}" already populated', context="Preprocessor")
        return [], 0.0

    metrics: list[dict[str, Any]] = []
    data_shape: tuple[int, ...] | None = None

    with atomic_directory(target_dir=out_folder) as staging_folder:
        metadata_path = staging_folder / "metadata"
        metadata_path.mkdir(parents=True, exist_ok=True)

        params_path = metadata_path / "params.json"
        metrics_path = metadata_path / "metrics.json"
        shape_path = metadata_path / "shape.csv"

        with params_path.open(mode="w", encoding="utf-8") as params_file:
            json.dump(preprocessing_option.to_params(), params_file, indent=2)

        for subject_id in track(
            iterable=range(1, 33),
            description=f"Preprocessing with option {{{preprocessing_option.name}}}",
            context="Preprocessor",
        ):
            start_time = time.perf_counter()
            out_path = _subject_npy_path(folder=staging_folder, subject_id=subject_id)
            raw = _load_raw_subject(subject_id=subject_id)
            data_out = preprocessing_option.preprocessing_method(raw, subject_id)
            data_shape = _ensure_shape(
                label="Subject data",
                shape_seen=data_shape,
                shape_new=tuple(data_out.shape),
            )

            np.save(file=out_path, arr=data_out)

            elapsed = time.perf_counter() - start_time
            metrics.append(
                {
                    "subject": subject_id,
                    "filename": out_path.name,
                    "seconds": elapsed,
                },
            )

        if data_shape is None:
            raise ValueError("No subject arrays were processed.")

        with metrics_path.open(mode="w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "total_seconds": float(sum(entry["seconds"] for entry in metrics)),
                    "subjects": metrics,
                },
                metrics_file,
                indent=2,
            )

        shape_fieldnames = [
            "subject_data_shape",
            "subjects",
        ]

        with shape_path.open(mode="w", encoding="utf-8", newline="") as shape_file:
            writer = csv.DictWriter(shape_file, fieldnames=shape_fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "subject_data_shape": _shape_to_str(data_shape),
                    "subjects": len(metrics),
                },
            )

    total_seconds = float(sum(entry["seconds"] for entry in metrics))
    return metrics, total_seconds


def _ensure_shape(
    *,
    label: str,
    shape_seen: tuple[int, ...] | None,
    shape_new: tuple[int, ...],
) -> tuple[int, ...]:
    """Track the first shape encountered and assert uniformity thereafter."""
    if shape_seen is None:
        return shape_new

    if shape_seen != shape_new:
        raise ValueError(f"{label} shapes are not uniform.")

    return shape_seen


def _split_trials(
    preprocessing_option: PreprocessingOption,
    *,
    subject_metrics: list[dict[str, Any]],
    subject_total_seconds: float,
) -> None:
    """
    Split subject-level arrays into per-trial joblib files atomically.
    """
    source_folder = preprocessing_option.get_subject_path()
    target_folder = preprocessing_option.get_trial_path()
    if directory_is_populated(path=target_folder):
        message(
            description=f'"{target_folder}" already populated', context="Preprocessor"
        )
        return

    npy_files = sorted(
        file_path for file_path in source_folder.iterdir() if file_path.suffix == ".npy"
    )
    ratings = pd.read_csv(filepath_or_buffer=DEAP_RATINGS_CSV)

    trial_counter = 0
    trial_metrics: list[dict[str, Any]] = []
    subject_data_shape: tuple[int, ...] | None = None
    trial_data_shape: tuple[int, ...] | None = None
    trial_dataframe_shape: tuple[int, ...] | None = None

    with atomic_directory(target_dir=target_folder) as staging_folder:
        metadata_path = staging_folder / "metadata"
        metadata_path.mkdir(parents=True, exist_ok=True)

        params_path = metadata_path / "params.json"
        metrics_path = metadata_path / "metrics.json"
        shape_path = metadata_path / "shape.csv"

        with params_path.open(mode="w", encoding="utf-8") as params_file:
            json.dump(preprocessing_option.to_params(), params_file, indent=2)

        for file_path in track(
            iterable=npy_files,
            description="Splitting subject into trials for "
            f"option {{{preprocessing_option.name}}}",
            context="Preprocessor",
        ):
            subject_id = int(file_path.stem[1:3])
            data = np.load(file=file_path)
            subject_data_shape = _ensure_shape(
                label="Subject data",
                shape_seen=subject_data_shape,
                shape_new=tuple(data.shape),
            )
            subj_mask = ratings["Participant_id"] == subject_id
            subj_ratings = cast("pd.DataFrame", ratings.loc[subj_mask]).sort_values(
                by="Experiment_id",
            )

            for trial_idx in range(TRIALS_NUM):
                start_time = time.perf_counter()
                trial_data = np.squeeze(a=data[trial_idx])
                trial_data_shape = _ensure_shape(
                    label="Trial data",
                    shape_seen=trial_data_shape,
                    shape_new=tuple(trial_data.shape),
                )
                row = subj_ratings.iloc[trial_idx]

                trial_df = pd.DataFrame(
                    data=[
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

                trial_dataframe_shape = _ensure_shape(
                    label="Trial dataframe",
                    shape_seen=trial_dataframe_shape,
                    shape_new=tuple(trial_df.shape),
                )

                trial_counter += 1
                out_name = f"t{trial_counter:04}.joblib"
                out_path = staging_folder / out_name
                joblib.dump(value=trial_df, filename=out_path, compress=3)

                elapsed = time.perf_counter() - start_time
                trial_metrics.append(
                    {
                        "subject": subject_id,
                        "trial_index": trial_idx,
                        "trial": out_name.removesuffix(".joblib"),
                        "filename": out_name,
                        "seconds": elapsed,
                    },
                )

        if trial_data_shape is None or trial_dataframe_shape is None:
            raise ValueError("No trials were generated; check preprocessing inputs.")

        trial_total_seconds = float(sum(entry["seconds"] for entry in trial_metrics))
        total_seconds = float(subject_total_seconds + trial_total_seconds)
        with metrics_path.open(mode="w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "total_seconds": total_seconds,
                    "subject_total_seconds": subject_total_seconds,
                    "trial_total_seconds": trial_total_seconds,
                    "subjects": subject_metrics,
                    "trials": trial_metrics,
                },
                metrics_file,
                indent=2,
            )

        if subject_data_shape is None:
            raise ValueError("No subject arrays were processed.")

        shape_fieldnames = [
            "subject_data_shape",
            "trial_data_shape",
            "trial_dataframe_shape",
            "subjects",
            "trials",
        ]

        with shape_path.open(mode="w", encoding="utf-8", newline="") as shape_file:
            writer = csv.DictWriter(shape_file, fieldnames=shape_fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "subject_data_shape": _shape_to_str(subject_data_shape),
                    "trial_data_shape": _shape_to_str(trial_data_shape),
                    "trial_dataframe_shape": _shape_to_str(trial_dataframe_shape),
                    "subjects": len(npy_files),
                    "trials": trial_counter,
                },
            )


def run_preprocessor(preprocessing_option: PreprocessingOption) -> None:
    """Execute both preprocessing stages for ``preprocessing_option``."""
    subject_metrics, subject_total_seconds = _preprocess_subjects(
        preprocessing_option=preprocessing_option,
    )
    _split_trials(
        preprocessing_option=preprocessing_option,
        subject_metrics=subject_metrics,
        subject_total_seconds=subject_total_seconds,
    )
