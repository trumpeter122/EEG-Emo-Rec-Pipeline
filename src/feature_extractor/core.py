"""Feature extraction pipeline for DEAP trials."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from config import BASELINE_SEC, SFREQ
from utils import message, track
from utils.fs import atomic_directory, directory_is_populated

if TYPE_CHECKING:
    from feature_extractor.types import FeatureExtractionOption

__all__ = ["run_feature_extractor"]


@dataclass(slots=True)
class _ExtractionSummary:
    """Shapes and counts describing a single trial extraction."""

    trial_data_shape: tuple[int, ...]
    baseline_shape: tuple[int, ...]
    segment_shape: tuple[int, ...]
    feature_dataframe_shape: tuple[int, int]
    segments: int


def _shape_to_str(shape: tuple[int, ...]) -> str:
    """Represent a numpy-like shape tuple as e.g. ``32x8064``."""
    return "x".join(str(dim) for dim in shape) if shape else "()"


def _extract_feature(
    trial_df: pd.DataFrame,
    feature_extraction_option: FeatureExtractionOption,
) -> tuple[pd.DataFrame, np.ndarray, _ExtractionSummary]:
    """
    Segment a trial into overlapping windows and compute features for each slice.

    - The baseline window always corresponds to the most recent segment that fits
      within the annotated baseline period.
    - All subsequent segments start after the baseline.
    """
    trial = trial_df.iloc[0]
    fe_method = feature_extraction_option.extraction_method

    segmentation_option = feature_extraction_option.segmentation_option
    window = int(segmentation_option.time_window * SFREQ)
    step = int(segmentation_option.time_step * SFREQ)

    trial_data = np.asarray(trial.get("data"))
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    _, n_samples = trial_data.shape
    if n_samples < window:
        raise ValueError("time_window is longer than the trial length.")

    baseline_samples = int(BASELINE_SEC * SFREQ)
    if window > baseline_samples:
        raise ValueError("time_window exceeds available baseline duration.")

    baseline_start = max(baseline_samples - window, 0)
    baseline_segment = trial_data[:, baseline_start : baseline_start + window]
    baseline_feature = fe_method(baseline_segment)

    trial_start = baseline_samples
    if trial_start >= n_samples:
        raise ValueError("trial segment starts beyond the available samples.")

    trial_samples = n_samples - baseline_samples
    if trial_samples < window:
        raise ValueError(
            "time_window is longer than the available trial duration after baseline.",
        )

    relative_starts = np.arange(
        start=0,
        stop=trial_samples - window + 1,
        step=step,
        dtype=int,
    )
    trial_starts = trial_start + relative_starts
    trial_features: list[np.ndarray] = []
    for start in trial_starts:
        segment = trial_data[:, start : start + window]
        trial_features.append(fe_method(segment))

    if not trial_features:
        raise ValueError(
            "No trial segments were generated; check time_window and time_step.",
        )

    out_df = pd.DataFrame(
        data={
            "data": [trial_features],
            **{col: trial[col] for col in trial.index if col != "data"},
        },
    ).explode(column=["data"], ignore_index=True)

    summary = _ExtractionSummary(
        trial_data_shape=tuple(trial_data.shape),
        baseline_shape=tuple(baseline_feature.shape),
        segment_shape=tuple(trial_features[0].shape),
        feature_dataframe_shape=(out_df.shape[0], out_df.shape[1]),
        segments=len(trial_features),
    )

    return out_df, baseline_feature, summary


def run_feature_extractor(
    feature_extraction_option: FeatureExtractionOption,
) -> None:
    """Extract baseline arrays and per-segment features atomically."""
    preprocessing_option = feature_extraction_option.preprocessing_option
    out_dir_path = feature_extraction_option.get_path()
    if directory_is_populated(path=out_dir_path):
        message(
            description=f'"{out_dir_path}" already populated',
            context="Feature Extractor",
        )
        return

    trials_path = preprocessing_option.get_trial_path()
    trial_files = sorted(trials_path.glob("*.joblib"))

    metrics: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []

    with atomic_directory(target_dir=out_dir_path) as staging_dir:
        metadata_path = staging_dir / "metadata"
        baseline_out_dir_path = metadata_path / "baseline"
        baseline_out_dir_path.mkdir(parents=True, exist_ok=True)

        params_path = metadata_path / "params.json"
        metrics_path = metadata_path / "metrics.json"
        shape_path = metadata_path / "shape.csv"

        with params_path.open(mode="w", encoding="utf-8") as params_file:
            json.dump(
                feature_extraction_option.to_params(),
                params_file,
                indent=2,
            )

        for trial_file in track(
            iterable=trial_files,
            description="Extracting feature with "
            f"option {{{feature_extraction_option.name}}} for "
            f"option {{{preprocessing_option.name}}}",
            context="Feature Extractor",
        ):
            out_path = staging_dir / f"{trial_file.stem}.joblib"
            baseline_out_path = baseline_out_dir_path / f"{trial_file.stem}.npy"

            trial_df = joblib.load(filename=trial_file)
            start_time = time.perf_counter()
            feature_df, baseline_array, summary = _extract_feature(
                trial_df=trial_df,
                feature_extraction_option=feature_extraction_option,
            )
            elapsed = time.perf_counter() - start_time

            np.save(file=baseline_out_path, arr=baseline_array)
            joblib.dump(value=feature_df, filename=out_path, compress=3)

            metrics.append(
                {
                    "trial": trial_file.stem,
                    "filename": trial_file.name,
                    "seconds": elapsed,
                },
            )
            shape_rows.append(
                {
                    "trial_data_shape": _shape_to_str(summary.trial_data_shape),
                    "baseline_shape": _shape_to_str(summary.baseline_shape),
                    "segment_shape": _shape_to_str(summary.segment_shape),
                    "feature_dataframe_shape": _shape_to_str(
                        summary.feature_dataframe_shape,
                    ),
                    "segments": summary.segments,
                },
            )

        total_seconds = float(sum(entry["seconds"] for entry in metrics))
        with metrics_path.open(mode="w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "total_seconds": total_seconds,
                    "trials": metrics,
                },
                metrics_file,
                indent=2,
            )

        shape_fieldnames = [
            "trial_data_shape",
            "baseline_shape",
            "segment_shape",
            "feature_dataframe_shape",
            "segments",
        ]

        first = shape_rows[0]
        if all(row == first for row in shape_rows[1:]):
            shape_row = [first]
        else:
            raise ValueError("Shapes are not uniform")

        with shape_path.open(mode="w", encoding="utf-8", newline="") as shape_file:
            writer = csv.DictWriter(shape_file, fieldnames=shape_fieldnames)
            writer.writeheader()
            writer.writerows(shape_row)
