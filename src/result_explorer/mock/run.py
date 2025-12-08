"""Simple runner for streaming real-time predictions from stored runs."""

from __future__ import annotations

import sys
from pathlib import Path

from config.constants import RESULTS_ROOT
from result_explorer.mock import (
    StreamingRequest,
    build_dual_pipeline,
    stream_dual_realtime_predictions,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Ensure `src/` is importable when executed via path (uv run src/result_explorer/mock/run.py)
ROOT = _repo_root()
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# Pipeline configuration (must be identical for arousal/valence except target)
PIPELINE_CONFIG = {
    "preprocessing": "unclean",
    "channel_pick": "minimal_frontopolar",
    "feature": "de",
    "segmentation": "w2.00_s0.25",
    "build_dataset": {
        "use_size": 1.0,
        "test_size": 0.3,
        "random_seed": 42,
        "target_kind": "classification_5",
        "feature_scaler": "standard",
    },
    "training_method": "adam_classification",
    "model": "cnn1d_n1_classification",
}

# Streaming controls
SEGMENT_PAUSE_SECONDS = 0.0
START_TRIAL_INDEX: int | None = None


def _build_feature_extraction_name() -> str:
    cfg = PIPELINE_CONFIG
    return "+".join(
        [
            cfg["preprocessing"],
            cfg["feature"],
            cfg["channel_pick"],
            cfg["segmentation"],
        ],
    )


def _build_dataset_name(*, target: str) -> str:
    ds_cfg = PIPELINE_CONFIG["build_dataset"]
    return "+".join(
        [
            target,
            f'use{ds_cfg["use_size"]:.2f}',
            f'test{ds_cfg["test_size"]:.2f}',
            f'seed{ds_cfg["random_seed"]}',
            f'{ds_cfg["target_kind"]}',
            f'{ds_cfg["feature_scaler"]}',
        ],
    )


def _build_training_data_name(*, target: str) -> str:
    return "+".join(
        [
            _build_feature_extraction_name(),
            _build_dataset_name(target=target),
        ],
    )


def _build_training_option_name(*, target: str) -> str:
    return "+".join(
        [
            _build_training_data_name(target=target),
            PIPELINE_CONFIG["training_method"],
        ],
    )


def _target_kind_dir() -> str:
    return str(PIPELINE_CONFIG["build_dataset"]["target_kind"])


def _run_matches(run_dir: Path, *, target: str) -> bool:
    try:
        params_path = run_dir / "params.json"
        params = params_path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        import json

        parsed = json.loads(params)
    except Exception:
        return False

    training_option = parsed.get("training_option", {})
    training_data_option = training_option.get("training_data_option", {})
    feature_extraction_option = training_data_option.get("feature_extraction_option", {})
    build_dataset_option = training_data_option.get("build_dataset_option", {})
    training_method_option = training_option.get("training_method_option", {})
    model_option = parsed.get("model_option", {})

    expected_feature_extraction = _build_feature_extraction_name()
    expected_build_dataset = _build_dataset_name(target=target)
    expected_training_data = _build_training_data_name(target=target)
    expected_training_option = _build_training_option_name(target=target)
    expected_training_method = PIPELINE_CONFIG["training_method"]
    expected_model = PIPELINE_CONFIG["model"]

    return all(
        [
            feature_extraction_option.get("name") == expected_feature_extraction,
            build_dataset_option.get("name") == expected_build_dataset,
            training_data_option.get("name") == expected_training_data,
            training_option.get("name") == expected_training_option,
            training_method_option.get("name") == expected_training_method,
            model_option.get("name") == expected_model,
        ],
    )


def _find_run_dir(*, target: str) -> Path:
    root = (ROOT / RESULTS_ROOT / target / _target_kind_dir()).resolve()
    if not root.exists():
        raise FileNotFoundError(f'No results directory found for target "{target}" at "{root}".')
    for run_dir in sorted(root.iterdir(), reverse=True):
        if run_dir.is_dir() and _run_matches(run_dir=run_dir, target=target):
            return run_dir
    raise FileNotFoundError(
        f'No run found matching the configured pipeline for target "{target}" under "{root}".',
    )


def _format_probabilities(probabilities: list[float] | None) -> str:
    if probabilities is None:
        return "-"
    formatted = ", ".join(f"{value:.3f}" for value in probabilities)
    return f"[{formatted}]"


def _run_streaming(
    *,
    arousal_run_dir: Path,
    valence_run_dir: Path,
    segment_pause_seconds: float,
    start_trial_index: int | None,
) -> None:
    if not arousal_run_dir.exists():
        raise FileNotFoundError(
            f'Arousal run directory "{arousal_run_dir}" does not exist.'
        )
    if not valence_run_dir.exists():
        raise FileNotFoundError(
            f'Valence run directory "{valence_run_dir}" does not exist.'
        )

    pipelines = build_dual_pipeline(
        arousal_run_dir=arousal_run_dir,
        valence_run_dir=valence_run_dir,
    )
    request = StreamingRequest(
        segment_pause_seconds=segment_pause_seconds,
        start_trial_index=start_trial_index,
    )

    for event in stream_dual_realtime_predictions(pipelines=pipelines, request=request):
        print(
            "subject="
            f"{event.subject} "
            f"trial={event.trial} "
            f"segment={event.segment_index} "
            f"t={event.segment_start_seconds:.2f}s "
            f"arousal={event.arousal_prediction:.4f} "
            f"valence={event.valence_prediction:.4f} "
            f"arousal_target={event.arousal_target if event.arousal_target is not None else '-'} "
            f"valence_target={event.valence_target if event.valence_target is not None else '-'} "
            f"arousal_prob={_format_probabilities(event.arousal_probabilities)} "
            f"valence_prob={_format_probabilities(event.valence_probabilities)}",
        )


def main() -> None:
    arousal_run_dir = _find_run_dir(target="arousal")
    valence_run_dir = _find_run_dir(target="valence")
    _run_streaming(
        arousal_run_dir=arousal_run_dir,
        valence_run_dir=valence_run_dir,
        segment_pause_seconds=SEGMENT_PAUSE_SECONDS,
        start_trial_index=START_TRIAL_INDEX,
    )


if __name__ == "__main__":
    main()
