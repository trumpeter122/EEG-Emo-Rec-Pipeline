"""Utilities for loading and ranking experiment results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config.constants import RESULTS_ROOT

TargetKind = Literal["classification", "regression"]

__all__ = [
    "TargetKind",
    "filter_results",
    "load_default_results_table",
    "load_results_table",
    "metric_sort_direction",
    "rank_runs",
]


@dataclass(frozen=True, slots=True)
class _RunLocation:
    target: str
    target_kind_dir: str
    run_dir: Path


def _load_json(path: Path) -> Any:
    """
    Load a UTF-8 JSON file.
    """
    with path.open(mode="r", encoding="utf-8") as source:
        return json.load(source)


def _repo_root() -> Path:
    """Return the repository root (two levels up from this module)."""
    return Path(__file__).resolve().parents[2]


def _resolve_results_root(*, base_dir: Path | None = None) -> Path:
    """
    Resolve RESULTS_ROOT relative to the repository root by default.

    - If RESULTS_ROOT is already absolute, return it unchanged.
    - Otherwise, anchor it at ``base_dir`` when provided or at the repo root.
    """
    root = RESULTS_ROOT
    if root.is_absolute():
        return root
    anchor = base_dir if base_dir is not None else _repo_root()
    return (anchor / root).resolve()


def _normalize_results_root(results_root: Path) -> Path:
    """
    Ensure the provided results_root is absolute and anchored to the repo.
    """
    if results_root.is_absolute():
        return results_root
    return (_repo_root() / results_root).resolve()


def _discover_run_locations(*, results_root: Path) -> list[_RunLocation]:
    """
    Enumerate run directories under the results root.
    """
    normalized_root = _normalize_results_root(results_root=results_root)
    if not normalized_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {normalized_root}")

    run_locations: list[_RunLocation] = []
    for target_dir in sorted(normalized_root.iterdir(), key=lambda path: path.name):
        if not target_dir.is_dir():
            continue
        for target_kind_dir in sorted(
            target_dir.iterdir(),
            key=lambda path: path.name,
        ):
            if not target_kind_dir.is_dir():
                continue
            for run_dir in sorted(
                target_kind_dir.iterdir(),
                key=lambda path: path.name,
            ):
                if run_dir.is_dir():
                    run_locations.append(
                        _RunLocation(
                            target=target_dir.name,
                            target_kind_dir=target_kind_dir.name,
                            run_dir=run_dir,
                        ),
                    )

    if not run_locations:
        raise FileNotFoundError(f"No runs found under {results_root}")

    return run_locations


def _extract_option_fields(*, params: dict[str, Any]) -> dict[str, Any]:
    """
    Pull option names and dataset attributes from the params payload.
    """
    training_option = params["training_option"]
    training_data_option = training_option["training_data_option"]
    feature_extraction_option = training_data_option["feature_extraction_option"]
    build_dataset_option = training_data_option["build_dataset_option"]
    preprocessing_option = feature_extraction_option["preprocessing_option"]
    feature_option = feature_extraction_option["feature_option"]
    channel_pick_option = feature_extraction_option["channel_pick_option"]
    segmentation_option = feature_extraction_option["segmentation_option"]
    training_method_option = training_option["training_method_option"]
    model_option = params["model_option"]

    return {
        "pipeline_name": params["name"],
        "training_option_name": training_option["name"],
        "training_data_option_name": training_data_option["name"],
        "feature_extraction_option_name": feature_extraction_option["name"],
        "preprocessing_option": preprocessing_option["name"],
        "feature_option": feature_option["name"],
        "channel_pick_option": channel_pick_option["name"],
        "segmentation_option": segmentation_option["name"],
        "build_dataset_option": build_dataset_option["name"],
        "model_option": model_option["name"],
        "training_method_option": training_method_option["name"],
        "feature_scaler": build_dataset_option["feature_scaler"],
        "random_seed": build_dataset_option["random_seed"],
        "use_size": build_dataset_option["use_size"],
        "test_size": build_dataset_option["test_size"],
        "target": build_dataset_option["target"],
        "target_kind": build_dataset_option["target_kind"],
    }


def _metric_key_from_kind(target_kind_value: str) -> TargetKind:
    """
    Map arbitrary target_kind strings to the canonical metric key.
    """
    lower = target_kind_value.lower()
    if "regress" in lower:
        return "regression"
    return "classification"


def _flatten_metric_values(
    *,
    metric_key: TargetKind,
    metrics: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    """
    Flatten numeric metrics under a prefix, ignoring array-like entries.
    """
    if metric_key not in metrics:
        raise KeyError(f"Missing expected metric group {metric_key}")

    flat_metrics: dict[str, float] = {}
    for metric_name, metric_value in metrics[metric_key].items():
        if isinstance(metric_value, bool):
            continue
        if isinstance(metric_value, (int, float)):
            flat_metrics[f"{prefix}_{metric_name}"] = float(metric_value)
    return flat_metrics


def _build_row(*, location: _RunLocation) -> dict[str, Any]:
    """
    Build a single DataFrame row from run artifacts.
    """
    params_path = location.run_dir / "params.json"
    metrics_path = location.run_dir / "metrics.json"
    params = _load_json(path=params_path)
    metrics = _load_json(path=metrics_path)

    option_fields = _extract_option_fields(params=params)
    if option_fields["target"] != location.target:
        raise ValueError("Target in params does not match directory layout.")
    metric_key = _metric_key_from_kind(option_fields["target_kind"])

    row: dict[str, Any] = {
        "target": location.target,
        "target_kind": option_fields["target_kind"],
        "target_kind_dir": location.target_kind_dir,
        "run_timestamp": metrics["run_timestamp"],
        "params_hash": metrics["params_hash"],
        "run_dir": str(location.run_dir),
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "total_seconds": metrics["total_seconds"],
        "epochs_logged": len(metrics["epochs"]),
    }
    row.update(option_fields)
    row.update(
        _flatten_metric_values(
            metric_key=metric_key,
            metrics=metrics["train_metrics"],
            prefix="train",
        ),
    )
    row.update(
        _flatten_metric_values(
            metric_key=metric_key,
            metrics=metrics["test_metrics"],
            prefix="test",
        ),
    )
    return row


def load_results_table(*, results_root: Path) -> pd.DataFrame:
    """
    Read all runs into a pandas DataFrame.
    """
    run_locations = _discover_run_locations(results_root=results_root)
    rows = [_build_row(location=location) for location in run_locations]
    return pd.DataFrame(rows)


def filter_results(
    *,
    results: pd.DataFrame,
    target: str,
    target_kind: str,
) -> pd.DataFrame:
    """
    Filter results DataFrame by target and target_kind.
    """
    kind_dir_mask = False
    if "target_kind_dir" in results.columns:
        kind_dir_mask = results["target_kind_dir"] == target_kind
    mask = (results["target"] == target) & (
        (results["target_kind"] == target_kind) | kind_dir_mask
    )
    return results.loc[mask].copy()


def metric_sort_direction(*, metric_name: str) -> Literal["asc", "desc"]:
    """
    Decide whether a metric should be sorted ascending or descending.
    """
    lower_is_better_tokens = ("loss", "mae", "mse", "rmse", "mape", "max_error")
    metric_name_lower = metric_name.lower()
    if any(token in metric_name_lower for token in lower_is_better_tokens):
        return "asc"
    return "desc"


def rank_runs(
    *,
    results: pd.DataFrame,
    metric_column: str,
    group_by_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    """
    Return the top_n runs per group for a given metric.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    missing_columns = [
        column
        for column in [metric_column, *group_by_columns]
        if column not in results.columns
    ]
    if missing_columns:
        missing_joined = ", ".join(missing_columns)
        raise KeyError(f"Missing columns: {missing_joined}")

    filtered = results.dropna(subset=[metric_column])
    sort_direction = metric_sort_direction(metric_name=metric_column)
    sorted_results = filtered.sort_values(
        by=metric_column,
        ascending=sort_direction == "asc",
        kind="mergesort",
    )
    if not group_by_columns:
        return sorted_results.head(top_n)
    grouped = sorted_results.groupby(group_by_columns, as_index=False, sort=False)
    return grouped.head(top_n)


def load_default_results_table() -> pd.DataFrame:
    """
    Convenience wrapper that loads results from the configured RESULTS_ROOT.
    """
    return load_results_table(results_root=_resolve_results_root())
