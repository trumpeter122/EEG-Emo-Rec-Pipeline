"""Generate report-ready summaries and plots from stored results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import textwrap
import pandas as pd

from feature_extractor.options import CHANNEL_PICK_OPTIONS
from result_explorer.results_viewer import (
    load_default_results_table,
    metric_sort_direction,
)

MetricName = str

OPTION_GROUPS: list[tuple[str, str, int]] = [
    ("Preprocessing", "preprocessing_option", 3),
    ("Feature", "feature_option", 6),
    ("Channel pick", "channel_pick_option", 6),
    ("Model", "model_option", 6),
]

PRIMARY_PURPLE = "#B39DDB"
PRIMARY_ORANGE = "#FFCC80"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _labelize(value: str) -> str:
    return value.replace("_", " ")


def _wrap_label(value: str, *, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(value, width=width))


def summarize_option_best(
    *,
    results: pd.DataFrame,
    target_kind_dir: str,
    option_column: str,
    metric: MetricName,
    top_n: int,
) -> pd.DataFrame:
    subset = results[results["target_kind_dir"] == target_kind_dir].dropna(
        subset=[metric],
    )
    per_target = (
        subset.groupby(["target", option_column], as_index=False)[metric]
        .max()
        .copy()
    )
    summary = (
        per_target.groupby(option_column, as_index=False)[metric]
        .mean()
        .copy()
    )
    ascending = metric_sort_direction(metric_name=metric) == "asc"
    summary = summary.sort_values(metric, ascending=ascending)
    if top_n > 0:
        summary = summary.head(top_n)
    return summary


def summarize_low_channel_best(
    *,
    results: pd.DataFrame,
    target_kind_dir: str,
    metric: MetricName,
    max_channels: int,
) -> pd.DataFrame:
    low_channel_names = [
        option.name
        for option in CHANNEL_PICK_OPTIONS
        if len(option.channel_pick) <= max_channels
    ]
    subset = results[
        (results["target_kind_dir"] == target_kind_dir)
        & (results["channel_pick_option"].isin(low_channel_names))
    ].dropna(subset=[metric])
    per_target = (
        subset.groupby(["target", "channel_pick_option"], as_index=False)[metric]
        .max()
        .copy()
    )
    summary = (
        per_target.groupby("channel_pick_option", as_index=False)[metric]
        .mean()
        .copy()
    )
    ascending = metric_sort_direction(metric_name=metric) == "asc"
    return summary.sort_values(metric, ascending=ascending)


def plot_option_summary(
    *,
    results: pd.DataFrame,
    target_kind_dir: str,
    metric: MetricName,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, option_column, top_n) in zip(axes.flat, OPTION_GROUPS):
        summary = summarize_option_best(
            results=results,
            target_kind_dir=target_kind_dir,
            option_column=option_column,
            metric=metric,
            top_n=top_n,
        )
        labels = [_wrap_label(_labelize(value)) for value in summary[option_column]]
        values = summary[metric].to_list()
        ax.barh(labels, values, color=PRIMARY_PURPLE)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.08)
        for index, value in enumerate(values):
            ax.text(
                min(value + 0.015, 0.98),
                index,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_low_channel_summary(
    *,
    results: pd.DataFrame,
    target_kind_dirs: list[str],
    metric: MetricName,
    max_channels: int,
    output_path: Path,
) -> None:
    summaries = []
    max_count = 0
    for target_kind_dir in target_kind_dirs:
        summary = summarize_low_channel_best(
            results=results,
            target_kind_dir=target_kind_dir,
            metric=metric,
            max_channels=max_channels,
        )
        summaries.append(summary)
        max_count = max(max_count, len(summary))

    height = max(4.2, 0.45 * max_count + 1.2)
    fig, axes = plt.subplots(
        1,
        len(target_kind_dirs),
        figsize=(11, height),
        constrained_layout=True,
    )
    if len(target_kind_dirs) == 1:
        axes = [axes]
    for ax, target_kind_dir, summary in zip(axes, target_kind_dirs, summaries):
        labels = [_wrap_label(_labelize(value)) for value in summary["channel_pick_option"]]
        values = summary[metric].to_list()
        ax.barh(labels, values, color=PRIMARY_ORANGE)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.08)
        for index, value in enumerate(values):
            ax.text(
                min(value + 0.015, 0.98),
                index,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_default_results_table()
    output_dir = _repo_root() / "report" / "typst" / "assets" / "results"
    plot_option_summary(
        results=results,
        target_kind_dir="classification",
        metric="test_f1_macro",
        output_path=output_dir / "options_summary_classification.png",
    )
    plot_option_summary(
        results=results,
        target_kind_dir="classification_3",
        metric="test_f1_macro",
        output_path=output_dir / "options_summary_classification_3.png",
    )
    plot_option_summary(
        results=results,
        target_kind_dir="classification_5",
        metric="test_f1_macro",
        output_path=output_dir / "options_summary_classification_5.png",
    )
    for max_channels in (4, 8, 12):
        plot_low_channel_summary(
            results=results,
            target_kind_dirs=["classification_3", "classification_5"],
            metric="test_f1_macro",
            max_channels=max_channels,
            output_path=output_dir / f"low_channel_summary_{max_channels}.png",
        )


if __name__ == "__main__":
    main()
