#!/usr/bin/env -S uvx --with numpy,pandas,matplotlib,seaborn python
# -*- coding: utf-8 -*-
"""Generate an HTML report for all runs under the results/ directory.

Usage
-----
From the project root:

    python scripts/report.py

or, if executable:

    ./scripts/report.py

By default it looks for ./results and writes ./results/report.html.

You can also pass a custom results directory:

    python scripts/report.py path/to/results

The script prints ONLY the (relative) path to the main HTML entry file on stdout,
so it can be used easily in shell pipelines.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunInfo:
    """Structured info about a single training run."""
    run_id: int
    run_dir: Path          # absolute path on disk (not shown in HTML)
    run_rel_dir: str       # path relative to results_dir, shown in HTML
    dataset_name: Optional[str]
    model_name: Optional[str]
    training_method_name: Optional[str]
    target: Optional[str]
    target_kind: Optional[str]
    feature_name: Optional[str]
    channel_pick: Optional[str]
    segmentation_name: Optional[str]
    timestamp: Optional[str]
    metrics_kind: Optional[str]  # "classification" / "regression"
    best_epoch: Optional[int]
    best_val_loss: Optional[float]
    total_seconds: Optional[float]
    summary_metrics: Dict[str, Any]
    has_confusion: bool


# ---------------------------------------------------------------------------
# Discovery / parsing
# ---------------------------------------------------------------------------


def find_runs(results_dir: Path) -> List[RunInfo]:
    """Find all runs by locating metrics.json + params.json pairs under results_dir."""
    runs: List[RunInfo] = []
    run_id = 0

    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        params_path = run_dir / "params.json"
        if not params_path.exists():
            continue

        try:
            with metrics_path.open("r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {metrics_path}: {e}", file=sys.stderr)
            continue

        try:
            with params_path.open("r") as f:
                params = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {params_path}: {e}", file=sys.stderr)
            continue

        try:
            run_rel_dir = str(run_dir.relative_to(results_dir))
        except ValueError:
            # Fallback, but still avoid absolute paths
            run_rel_dir = run_dir.name

        mtop = metrics
        ptop = params

        m_option = ptop.get("model_training_option", {}).get("model_option", {})
        t_option = ptop.get("model_training_option", {}).get("training_option", {})
        data_option = t_option.get("training_data_option", {})
        feature_opt = data_option.get("feature_extraction_option", {})
        channel_pick_opt = feature_opt.get("channel_pick_option", {})
        seg_opt = feature_opt.get("segmentation_option", {})

        model_name = m_option.get("name")
        target_kind = m_option.get("target_kind")
        training_method_opt = t_option.get("training_method_option", {})
        training_method_name = training_method_opt.get("name")

        dataset_name = feature_opt.get("name") or data_option.get("name")
        feature_name = feature_opt.get("feature_option", {}).get("name")
        channel_pick = channel_pick_opt.get("name")
        segmentation_name = seg_opt.get("name")
        target = data_option.get("target")
        timestamp = ptop.get("time stamp")  # already a string

        best_epoch = mtop.get("best_epoch")
        best_val_loss = mtop.get("best_val_loss")
        total_seconds = mtop.get("total_seconds")

        test_metrics = (mtop.get("test_metrics") or {})

        if "classification" in test_metrics:
            metrics_kind = "classification"
        elif "regression" in test_metrics:
            metrics_kind = "regression"
        else:
            metrics_kind = None

        summary: Dict[str, Any] = {}
        has_confusion = False

        if metrics_kind == "classification":
            test_cls = test_metrics.get("classification", {})
            for key in [
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
            ]:
                if key in test_cls:
                    summary[f"test_{key}"] = test_cls.get(key)
            if "confusion_matrix" in test_cls:
                has_confusion = True

        elif metrics_kind == "regression":
            test_reg = test_metrics.get("regression", {})
            for key in ["mae", "mse", "rmse", "r2"]:
                if key in test_reg:
                    summary[f"test_{key}"] = test_reg.get(key)

        runs.append(
            RunInfo(
                run_id=run_id,
                run_dir=run_dir,
                run_rel_dir=run_rel_dir,
                dataset_name=dataset_name,
                model_name=model_name,
                training_method_name=training_method_name,
                target=target,
                target_kind=target_kind,
                feature_name=feature_name,
                channel_pick=channel_pick,
                segmentation_name=segmentation_name,
                timestamp=timestamp,
                metrics_kind=metrics_kind,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                total_seconds=total_seconds,
                summary_metrics=summary,
                has_confusion=has_confusion,
            )
        )
        run_id += 1

    return runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_label(r: RunInfo) -> str:
    """Human-readable label for sidebar and run card headers."""
    parts: List[str] = []
    if r.dataset_name:
        parts.append(r.dataset_name)
    if r.model_name:
        parts.append(r.model_name)
    if r.training_method_name:
        parts.append(r.training_method_name)
    label = " | ".join(parts) if parts else r.run_rel_dir
    if r.target:
        label = f"{label} ({r.target})"
    return label


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_learning_curves(metrics: Dict[str, Any], title: str) -> Optional[str]:
    """Learning curves from metrics['epochs']."""
    epochs = metrics.get("epochs", [])
    if not epochs:
        return None

    df = pd.DataFrame(epochs)
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    if "train_loss" in df.columns:
        sns.lineplot(data=df, x="epoch", y="train_loss", ax=ax, label="train_loss")
    if "val_loss" in df.columns:
        sns.lineplot(data=df, x="epoch", y="val_loss", ax=ax, label="val_loss")
    if "train_mae" in df.columns:
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_mae",
            ax=ax,
            label="train_mae",
            linestyle="--",
        )
    if "val_mae" in df.columns:
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_mae",
            ax=ax,
            label="val_mae",
            linestyle="--",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig_to_base64(fig)


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    split: str,
    title: str,
) -> Optional[str]:
    """Confusion matrix heatmap for a given split."""
    mt = metrics.get(f"{split}_metrics", {})
    cls = mt.get("classification", {})
    cm = cls.get("confusion_matrix")
    labels = cls.get("labels")
    if cm is None or labels is None:
        return None

    cm_arr = np.array(cm)
    labels_arr = np.array(labels)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm_arr,
        ax=ax,
        cmap="viridis",
        cbar=True,
        square=True,
        xticklabels=labels_arr,
        yticklabels=labels_arr,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title} ({split})")
    fig.tight_layout()

    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Summary dataframe + overview plot
# ---------------------------------------------------------------------------


def build_summary_dataframe(runs: List[RunInfo]) -> pd.DataFrame:
    """Flatten RunInfo list into a summary DataFrame."""
    rows: List[Dict[str, Any]] = []
    for r in runs:
        row = {
            "run_id": r.run_id,
            "run_dir": r.run_rel_dir,  # relative only
            "dataset": r.dataset_name,
            "feature": r.feature_name,
            "channels": r.channel_pick,
            "segmentation": r.segmentation_name,
            "model": r.model_name,
            "training_method": r.training_method_name,
            "target": r.target,
            "target_kind": r.target_kind,
            "metrics_kind": r.metrics_kind,
            "best_epoch": r.best_epoch,
            "best_val_loss": r.best_val_loss,
            "total_seconds": r.total_seconds,
            "timestamp": r.timestamp,
        }
        row.update(r.summary_metrics)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def sort_runs_by_performance(
    runs: List[RunInfo],
    summary_df: pd.DataFrame,
    kind: str,
) -> List[RunInfo]:
    """Sort runs by performance: accuracy desc for classification, MAE asc for regression."""
    if summary_df.empty:
        return []

    df = summary_df[summary_df["metrics_kind"] == kind].copy()
    if df.empty:
        return []

    if kind == "classification":
        if "test_accuracy" in df.columns:
            df["score"] = df["test_accuracy"]
        elif "test_f1_macro" in df.columns:
            df["score"] = df["test_f1_macro"]
        else:
            df["score"] = 0.0
        ascending = False  # higher is better
    else:  # regression
        if "test_mae" in df.columns:
            df["score"] = df["test_mae"]
        elif "test_rmse" in df.columns:
            df["score"] = df["test_rmse"]
        elif "best_val_loss" in df.columns:
            df["score"] = df["best_val_loss"]
        else:
            df["score"] = np.inf
        ascending = True  # lower is better

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df.sort_values("score", ascending=ascending, inplace=True, na_position="last")

    id_to_run = {r.run_id: r for r in runs}
    sorted_runs: List[RunInfo] = []
    for rid in df["run_id"].tolist():
        r = id_to_run.get(rid)
        if r is not None:
            sorted_runs.append(r)
    return sorted_runs


def plot_overall_bar(summary_df: pd.DataFrame, kind: str) -> Optional[str]:
    """Bar chart of primary metric across runs (accuracy for classification, MAE for regression)."""
    if summary_df.empty:
        return None

    df = summary_df[summary_df["metrics_kind"] == kind].copy()
    if df.empty:
        return None

    if kind == "classification":
        metric_col = "test_accuracy" if "test_accuracy" in df.columns else None
        ylabel = "Test accuracy"
        title = "Classification runs: test accuracy by configuration"
    else:
        metric_col = "test_mae" if "test_mae" in df.columns else None
        ylabel = "Test MAE"
        title = "Regression runs: test MAE by configuration"

    if metric_col is None or metric_col not in df.columns:
        return None

    df = df.dropna(subset=[metric_col])
    if df.empty:
        return None

    df["label"] = (
        df["dataset"].fillna("?")
        + " | "
        + df["model"].fillna("?")
        + " | "
        + df["training_method"].fillna("?")
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="label", y=metric_col, ax=ax)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    ax.set_title(title)
    # Skip tight_layout here to avoid warnings with many x labels

    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML building helpers
# ---------------------------------------------------------------------------


def df_to_html_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as an HTML table with our CSS classes."""
    return df.to_html(
        index=False,
        classes=["summary-table"],
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        border=0,
    )


def build_sidebar_html(
    runs_sorted: List[RunInfo],
    summary_df: pd.DataFrame,
    kind: str,
) -> str:
    """Sidebar links for quick navigation."""
    if not runs_sorted:
        return "<div class='sidebar-empty'>No runs found.</div>"

    df_idx = summary_df.set_index("run_id")
    items: List[str] = []

    for r in runs_sorted:
        label = run_label(r)
        metric_display = ""
        if r.run_id in df_idx.index:
            row = df_idx.loc[r.run_id]
            if kind == "classification" and "test_accuracy" in row and pd.notna(row["test_accuracy"]):
                metric_display = f"acc: {row['test_accuracy']:.4f}"
            elif kind == "regression" and "test_mae" in row and pd.notna(row["test_mae"]):
                metric_display = f"MAE: {row['test_mae']:.4f}"
        items.append(
            f"""
            <a href="#run-{r.run_id}" class="sidebar-link">
              <div class="sidebar-run-name">{label}</div>
              <div class="sidebar-run-metric">{metric_display}</div>
            </a>
            """
        )

    return "\n".join(items)


def build_run_cards_html(
    runs_sorted: List[RunInfo],
    summary_df: pd.DataFrame,
) -> str:
    """Full HTML for all run cards, ordered by performance."""
    if not runs_sorted:
        return ""

    sections: List[str] = []
    for r in runs_sorted:
        metrics_path = r.run_dir / "metrics.json"
        try:
            with metrics_path.open("r") as f:
                mtop = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to reload metrics for {metrics_path}: {e}", file=sys.stderr)
            continue

        label = run_label(r)

        # Plots (learning curves + confusion matrices if available)
        lc_b64 = plot_learning_curves(mtop, title=f"{label} — learning curves")
        cm_train_b64 = (
            plot_confusion_matrix(mtop, split="train", title=label)
            if r.metrics_kind == "classification"
            else None
        )
        cm_test_b64 = (
            plot_confusion_matrix(mtop, split="test", title=label)
            if r.metrics_kind == "classification"
            else None
        )

        meta_items = [
            ("Run directory", r.run_rel_dir),
            ("Dataset", r.dataset_name),
            ("Feature", r.feature_name),
            ("Channels", r.channel_pick),
            ("Segmentation", r.segmentation_name),
            ("Model", r.model_name),
            ("Training method", r.training_method_name),
            ("Target", r.target),
            ("Target kind", r.target_kind),
            ("Metrics kind", r.metrics_kind),
            ("Best epoch", r.best_epoch),
            ("Best val loss", r.best_val_loss),
            ("Training time (s)", r.total_seconds),
            ("Timestamp", r.timestamp),
        ]

        meta_html_items = "".join(
            f"<div class='meta-item'><span class='meta-key'>{k}</span>"
            f"<span class='meta-value'>{v}</span></div>"
            for k, v in meta_items
            if v is not None
        )

        summary_metric_html_items = ""
        for k, v in r.summary_metrics.items():
            if isinstance(v, (int, float)):
                v_str = f"{v:.4f}"
            else:
                v_str = str(v)
            summary_metric_html_items += (
                f"<div class='metric-item'><span class='metric-key'>{k}</span>"
                f"<span class='metric-value'>{v_str}</span></div>"
            )

        plots_html_inner = ""
        if lc_b64:
            plots_html_inner += (
                "<div class='plot-card'>"
                "<div class='plot-title'>Learning curves</div>"
                f"<img src='data:image/png;base64,{lc_b64}' alt='learning curves' />"
                "</div>"
            )
        if cm_train_b64:
            plots_html_inner += (
                "<div class='plot-card'>"
                "<div class='plot-title'>Confusion matrix (train)</div>"
                f"<img src='data:image/png;base64,{cm_train_b64}' alt='train confusion matrix' />"
                "</div>"
            )
        if cm_test_b64:
            plots_html_inner += (
                "<div class='plot-card'>"
                "<div class='plot-title'>Confusion matrix (test)</div>"
                f"<img src='data:image/png;base64,{cm_test_b64}' alt='test confusion matrix' />"
                "</div>"
            )

        plots_html = ""
        if plots_html_inner:
            # Put plots on their own full-width row
            plots_html = (
                "<div class='run-plots-row'>"
                "<div class='run-plots'>"
                f"{plots_html_inner}"
                "</div>"
                "</div>"
            )

        section_html = (
            f"<section class='run-card' id='run-{r.run_id}'>"
            "<header class='run-header'>"
            f"<h2>{label}</h2>"
            f"<div class='run-subtitle'>Dataset: {r.dataset_name or 'N/A'} | "
            f"Target: {r.target or 'N/A'} | Metrics: {r.metrics_kind or 'N/A'}</div>"
            "</header>"
            "<div class='run-body'>"
            "<div class='run-meta'>"
            f"{meta_html_items}"
            "</div>"
            "<div class='run-metrics'>"
            f"{summary_metric_html_items}"
            "</div>"
            "</div>"
            f"{plots_html}"
            "</section>"
        )

        sections.append(section_html)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def generate_html_report(results_dir: Path, output_path: Path) -> bool:
    """Scan results_dir and write a complete HTML report to output_path."""
    runs = find_runs(results_dir)
    if not runs:
        print(f"No runs found under {results_dir}", file=sys.stderr)
        return False

    summary_df = build_summary_dataframe(runs)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_label = results_dir.name or "results"

    # Split by kind
    summary_cls = summary_df[summary_df["metrics_kind"] == "classification"].copy()
    summary_reg = summary_df[summary_df["metrics_kind"] == "regression"].copy()

    runs_cls_sorted = sort_runs_by_performance(runs, summary_df, "classification")
    runs_reg_sorted = sort_runs_by_performance(runs, summary_df, "regression")

    # Overview bar plots (per kind)
    overall_bar_cls_b64 = plot_overall_bar(summary_df, "classification")
    overall_bar_reg_b64 = plot_overall_bar(summary_df, "regression")

    # Tables
    cls_table_html = (
        df_to_html_table(summary_cls) if not summary_cls.empty else "<div>No classification runs.</div>"
    )
    reg_table_html = (
        df_to_html_table(summary_reg) if not summary_reg.empty else "<div>No regression runs.</div>"
    )

    # Sidebars
    sidebar_cls_html = build_sidebar_html(runs_cls_sorted, summary_df, "classification")
    sidebar_reg_html = build_sidebar_html(runs_reg_sorted, summary_df, "regression")

    # Run cards
    per_run_cls_html = build_run_cards_html(runs_cls_sorted, summary_df)
    per_run_reg_html = build_run_cards_html(runs_reg_sorted, summary_df)

    # Overview sections
    cls_overview_html = ""
    if overall_bar_cls_b64:
        cls_overview_html = (
            "<section class='section-card'>"
            "<h2>Classification overview</h2>"
            "<div class='plot-card wide'>"
            f"<img src='data:image/png;base64,{overall_bar_cls_b64}' alt='classification summary' />"
            "</div>"
            "</section>"
        )

    reg_overview_html = ""
    if overall_bar_reg_b64:
        reg_overview_html = (
            "<section class='section-card'>"
            "<h2>Regression overview</h2>"
            "<div class='plot-card wide'>"
            f"<img src='data:image/png;base64,{overall_bar_reg_b64}' alt='regression summary' />"
            "</div>"
            "</section>"
        )

    total_runs = len(runs)
    total_datasets = summary_df["dataset"].nunique() if not summary_df.empty else 0
    total_models = summary_df["model"].nunique() if not summary_df.empty else 0
    total_cls = len(runs_cls_sorted)
    total_reg = len(runs_reg_sorted)

    # Main HTML frame with top navbar and left sidebar
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Training report</title>
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
    }}
    body {{
      font-family: "JetBrains Mono", SFMono-Regular, Menlo, Monaco, Consolas,
                   "Liberation Mono", "Courier New", monospace;
      margin: 0;
      padding: 0;
      background: #f5f5f5;
      color: #111827;
      line-height: 1.4;
    }}
    code {{
      font-family: inherit;
    }}
    .page {{
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}
    .page-inner {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 12px 10px 24px 10px;
      flex: 1;
      display: flex;
      flex-direction: column;
    }}
    header {{
      margin-bottom: 8px;
    }}
    h1 {{
      font-size: 1.9rem;
      margin-bottom: 0.25rem;
      font-weight: 700;
    }}
    h2 {{
      font-size: 1.1rem;
      margin: 0;
      font-weight: 600;
    }}
    .subtitle {{
      color: #4b5563;
      font-size: 0.8rem;
      margin-bottom: 0.75rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .pill {{
      border-radius: 999px;
      border: 1px solid #d4d4d8;
      padding: 2px 8px;
      font-size: 0.74rem;
      color: #111827;
      background: #ffffff;
    }}
    .top-nav {{
      display: inline-flex;
      border-radius: 999px;
      background: #e5e7eb;
      border: 1px solid #d4d4d8;
      overflow: hidden;
      margin-top: 4px;
    }}
    .nav-tab {{
      border: none;
      background: transparent;
      color: #4b5563;
      padding: 4px 12px;
      font-size: 0.8rem;
      cursor: pointer;
      outline: none;
    }}
    .nav-tab.active {{
      background: #ffffff;
      color: #111827;
      border-radius: 999px;
      box-shadow: 0 0 0 1px #d4d4d8;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr);
      gap: 12px;
      flex: 1;
      min-height: 0;
      margin-top: 6px;
    }}
    @media (max-width: 960px) {{
      .layout {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}

    .sidebar {{
      background: #ffffff;
      border-radius: 10px;
      padding: 8px;
      box-shadow: 0 10px 18px rgba(15,23,42,0.06);
      border: 1px solid #e5e7eb;
      overflow-y: auto;
      max-height: calc(100vh - 130px);
      font-size: 0.75rem;
    }}
    .sidebar-title {{
      font-size: 0.76rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 6px;
    }}
    .sidebar-link {{
      display: block;
      padding: 6px 7px;
      border-radius: 7px;
      text-decoration: none;
      color: inherit;
      margin-bottom: 4px;
      background: #f9fafb;
      border: 1px solid transparent;
      transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
    }}
    .sidebar-link:hover {{
      background: #eff6ff;
      border-color: #bfdbfe;
      transform: translateX(1px);
    }}
    .sidebar-run-name {{
      font-size: 0.73rem;
      font-weight: 500;
      word-break: break-all;
      overflow-wrap: anywhere;
    }}
    .sidebar-run-metric {{
      font-size: 0.7rem;
      color: #6b7280;
    }}
    .sidebar-empty {{
      font-size: 0.75rem;
      color: #6b7280;
    }}
    .sidebar-section {{
      display: none;
    }}
    .sidebar-section.active {{
      display: block;
    }}

    .content {{
      overflow-y: auto;
      max-height: calc(100vh - 130px);
      padding-right: 2px;
    }}

    .section-card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 12px 16px 14px 16px;
      box-shadow: 0 10px 18px rgba(15,23,42,0.06);
      margin-bottom: 16px;
      border: 1px solid #e5e7eb;
      font-size: 0.8rem;
    }}

    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      table-layout: fixed;
    }}
    .summary-table th,
    .summary-table td {{
      padding: 4px 6px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      vertical-align: top;
      word-break: break-all;
      overflow-wrap: anywhere;
    }}
    .summary-table th {{
      background: #f3f4f6;
      font-weight: 600;
    }}

    .run-card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 12px 16px 14px 16px;
      box-shadow: 0 10px 18px rgba(15,23,42,0.06);
      margin-bottom: 16px;
      border: 1px solid #e5e7eb;
      font-size: 0.8rem;
    }}
    .run-header {{
      margin-bottom: 8px;
    }}
    .run-subtitle {{
      font-size: 0.76rem;
      color: #4b5563;
      margin-top: 3px;
      word-break: break-all;
      overflow-wrap: anywhere;
    }}
    .run-body {{
      display: grid;
      grid-template-columns: minmax(0, 240px) minmax(0, 1fr);
      gap: 10px;
      align-items: flex-start;
    }}
    @media (max-width: 960px) {{
      .run-body {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}

    .run-meta {{
      font-size: 0.76rem;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 3px;
    }}
    .meta-item {{
      display: flex;
      justify-content: space-between;
      gap: 6px;
      border-bottom: 1px dashed #e5e7eb;
      padding: 2px 0;
    }}
    .meta-key {{
      color: #6b7280;
      flex: 0 0 auto;
      max-width: 40%;
    }}
    .meta-value {{
      color: #111827;
      text-align: right;
      word-break: break-all;
      overflow-wrap: anywhere;
      flex: 1 1 auto;
    }}

    .run-metrics {{
      font-size: 0.76rem;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 4px 6px;
    }}
    .metric-item {{
      background: #f9fafb;
      border-radius: 7px;
      padding: 4px 6px;
      border: 1px solid #e5e7eb;
    }}
    .metric-key {{
      display: block;
      color: #6b7280;
      font-size: 0.7rem;
    }}
    .metric-value {{
      font-weight: 600;
      font-size: 0.82rem;
    }}

    .run-plots-row {{
      margin-top: 10px;
    }}
    .run-plots {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
    }}
    .plot-card {{
      background: #f9fafb;
      border-radius: 8px;
      padding: 6px;
      border: 1px solid #e5e7eb;
    }}
    .plot-card.wide img {{
      width: 100%;
      height: auto;
    }}
    .plot-title {{
      font-size: 0.75rem;
      color: #4b5563;
      margin-bottom: 4px;
    }}

    img {{
      max-width: 100%;
      display: block;
      border-radius: 5px;
    }}

    .section-group {{
      display: none;
    }}
    .section-group.active {{
      display: block;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="page-inner">
      <header>
        <h1>Training report</h1>
        <div class="subtitle">Generated at {now_str}. Results directory: <code>{results_label}</code></div>
        <div class="pill-row">
          <span class="pill">Total runs: {total_runs}</span>
          <span class="pill">Datasets: {total_datasets}</span>
          <span class="pill">Models: {total_models}</span>
          <span class="pill">Classification: {total_cls}</span>
          <span class="pill">Regression: {total_reg}</span>
        </div>
        <div class="top-nav">
          <button class="nav-tab active" data-target="classification">Classification</button>
          <button class="nav-tab" data-target="regression">Regression</button>
        </div>
      </header>

      <div class="layout">
        <nav class="sidebar">
          <div id="sidebar-classification" class="sidebar-section active">
            <div class="sidebar-title">Classification runs (sorted by accuracy)</div>
            {sidebar_cls_html}
          </div>
          <div id="sidebar-regression" class="sidebar-section">
            <div class="sidebar-title">Regression runs (sorted by MAE)</div>
            {sidebar_reg_html}
          </div>
        </nav>

        <main class="content">
          <div id="section-classification" class="section-group active">
            <section class="section-card">
              <h2>Classification summary</h2>
              <div style="max-height: 320px; overflow: auto; margin-top: 6px;">
                {cls_table_html}
              </div>
            </section>
            {cls_overview_html}
            {per_run_cls_html}
          </div>

          <div id="section-regression" class="section-group">
            <section class="section-card">
              <h2>Regression summary</h2>
              <div style="max-height: 320px; overflow: auto; margin-top: 6px;">
                {reg_table_html}
              </div>
            </section>
            {reg_overview_html}
            {per_run_reg_html}
          </div>
        </main>
      </div>
    </div>
  </div>

  <script>
    (function() {{
      var tabs = document.querySelectorAll('.nav-tab');
      var sectionCls = document.getElementById('section-classification');
      var sectionReg = document.getElementById('section-regression');
      var sidebarCls = document.getElementById('sidebar-classification');
      var sidebarReg = document.getElementById('sidebar-regression');

      function activate(target) {{
        tabs.forEach(function(btn) {{
          if (btn.dataset.target === target) {{
            btn.classList.add('active');
          }} else {{
            btn.classList.remove('active');
          }}
        }});

        if (target === 'classification') {{
          sectionCls.classList.add('active');
          sectionReg.classList.remove('active');
          sidebarCls.classList.add('active');
          sidebarReg.classList.remove('active');
        }} else {{
          sectionCls.classList.remove('active');
          sectionReg.classList.add('active');
          sidebarCls.classList.remove('active');
          sidebarReg.classList.add('active');
        }}
      }}

      tabs.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          activate(btn.dataset.target);
        }});
      }});

      // Initial state
      activate('classification');
    }})();
  </script>
</body>
</html>
"""

    output_path.write_text(full_html, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str]) -> None:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    results_dir = root_dir / "results"

    # Optional custom results directory
    if len(argv) > 1:
        results_dir = Path(argv[1]).expanduser().resolve()

    if not results_dir.exists():
        print(f"results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = results_dir / "report.html"

    ok = generate_html_report(results_dir, output_path)
    if not ok:
        sys.exit(1)

    # Print ONLY a relative path on stdout for shell piping (avoid absolute paths)
    cwd = Path(os.getcwd())
    try:
        display_path = output_path.relative_to(cwd)
    except ValueError:
        display_path = output_path  # fallback, but usually under cwd

    print(str(display_path))


if __name__ == "__main__":
    main(sys.argv)
