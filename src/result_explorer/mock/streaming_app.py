"""Mock real-time prediction pipeline built on existing training artifacts."""

from __future__ import annotations

import importlib
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore[import-untyped]

from config.constants import BASELINE_SEC
from feature_extractor.core import _extract_feature
from feature_extractor.options import FEATURE_OPTIONS
from feature_extractor.types import (
    ChannelPickOption,
    FeatureExtractionOption,
    FeatureOption,
    SegmentationOption,
)
from model_trainer.options import MODEL_OPTIONS
from model_trainer.types import BuildDatasetOption, ModelOption
from preprocessor.options import PREPROCESSING_OPTIONS
from preprocessor.types import PreprocessingOption

from model_trainer.options.options_training_method.utils import _ensure_conv1d_shape

__all__ = [
    "RunArtifacts",
    "PipelineContext",
    "StreamingRequest",
    "SegmentPayload",
    "PredictionEvent",
    "CombinedPredictionEvent",
    "CombinedPipelines",
    "RealtimePredictor",
    "CombinedRealtimePredictor",
    "build_mock_pipeline",
    "build_dual_pipeline",
    "stream_realtime_predictions",
    "stream_dual_realtime_predictions",
]


@dataclass(slots=True)
class RunArtifacts:
    """Metadata parsed from a stored training run."""

    run_dir: Path
    params: dict[str, Any]
    build_dataset_params: dict[str, Any]
    target_name: str
    target_kind_raw: str
    target_kind: Literal["classification", "regression"]
    backend: Literal["torch", "sklearn"]
    class_labels: np.ndarray | None


@dataclass(slots=True)
class PipelineContext:
    """Reconstructed pipeline options for inference."""

    preprocessing_option: PreprocessingOption
    feature_extraction_option: FeatureExtractionOption
    build_dataset_option: BuildDatasetOption
    model_option: ModelOption
    artifacts: RunArtifacts


@dataclass(slots=True)
class CombinedPipelines:
    """Paired arousal/valence pipelines that share feature extraction."""

    arousal: PipelineContext
    valence: PipelineContext

    def __post_init__(self) -> None:
        arousal_fe = self.arousal.feature_extraction_option
        valence_fe = self.valence.feature_extraction_option
        if arousal_fe.name != valence_fe.name:
            raise ValueError(
                "Arousal and valence pipelines must use the same feature extraction configuration.",
            )
        if arousal_fe.preprocessing_option.name != valence_fe.preprocessing_option.name:
            raise ValueError(
                "Arousal and valence pipelines must share the same preprocessing option.",
            )
        if arousal_fe.segmentation_option.name != valence_fe.segmentation_option.name:
            raise ValueError(
                "Arousal and valence pipelines must share the same segmentation option.",
            )
        arousal_ds = self.arousal.artifacts.build_dataset_params
        valence_ds = self.valence.artifacts.build_dataset_params
        _ensure_build_dataset_match(arousal_ds, valence_ds)
        _ensure_training_and_model_match(
            arousal_params=self.arousal.artifacts.params,
            valence_params=self.valence.artifacts.params,
        )


@dataclass(slots=True)
class StreamingRequest:
    """User-provided knobs for the streaming demo."""

    segment_pause_seconds: float
    start_trial_index: int | None = None


@dataclass(slots=True)
class SegmentPayload:
    """Raw feature tensor + metadata emitted by the stream."""

    feature: np.ndarray
    subject: int
    trial: int
    segment_index: int
    segment_start_seconds: float
    targets: dict[str, float | None]


@dataclass(slots=True)
class PredictionEvent:
    """Model output paired with the originating segment metadata."""

    prediction: float
    subject: int
    trial: int
    segment_index: int
    segment_start_seconds: float
    target_value: float | None
    probabilities: list[float] | None


@dataclass(slots=True)
class CombinedPredictionEvent:
    """Simultaneous arousal/valence outputs for a shared segment."""

    subject: int
    trial: int
    segment_index: int
    segment_start_seconds: float
    arousal_prediction: float
    valence_prediction: float
    arousal_target: float | None
    valence_target: float | None
    arousal_probabilities: list[float] | None
    valence_probabilities: list[float] | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(mode="r", encoding="utf-8") as source:
        return json.load(fp=source)


def _import_attr(dotted_path: str) -> Callable[..., Any]:
    module_path, attr_name = dotted_path.rsplit(sep=".", maxsplit=1)
    module = importlib.import_module(name=module_path)
    return getattr(module, attr_name)


def _normalize_target_kind(raw_kind: str) -> Literal["classification", "regression"]:
    lower = raw_kind.lower()
    if "class" in lower:
        return "classification"
    return "regression"


def _class_labels_array(build_dataset_params: dict[str, Any]) -> np.ndarray | None:
    labels = build_dataset_params.get("class_labels")
    if labels is None:
        return None
    return np.asarray(a=labels, dtype=np.float32)


def _build_dataset_signature(params: dict[str, Any]) -> tuple[Any, ...]:
    return (
        params.get("random_seed"),
        params.get("use_size"),
        params.get("test_size"),
        params.get("target_kind"),
        params.get("feature_scaler"),
    )


def _ensure_build_dataset_match(
    arousal_ds: dict[str, Any],
    valence_ds: dict[str, Any],
) -> None:
    if arousal_ds.get("target") == valence_ds.get("target"):
        raise ValueError("Arousal and valence targets must be different.")
    if _build_dataset_signature(arousal_ds) != _build_dataset_signature(valence_ds):
        raise ValueError(
            "Arousal and valence build_dataset configurations must match except for the target.",
        )


def _ensure_training_and_model_match(
    *,
    arousal_params: dict[str, Any],
    valence_params: dict[str, Any],
) -> None:
    def _method_name(payload: dict[str, Any]) -> str | None:
        return (
            payload.get("training_option", {})
            .get("training_method_option", {})
            .get("name")
        )

    def _model_name(payload: dict[str, Any]) -> str | None:
        return payload.get("model_option", {}).get("name")

    if _method_name(arousal_params) != _method_name(valence_params):
        raise ValueError(
            "Training method options must match for arousal and valence pipelines.",
        )
    if _model_name(arousal_params) != _model_name(valence_params):
        raise ValueError(
            "Model options must match for arousal and valence pipelines.",
        )


def _resolve_preprocessing_option(name: str) -> PreprocessingOption:
    try:
        return PREPROCESSING_OPTIONS.get_name(name=name)
    except KeyError as exc:
        raise RuntimeError(
            f'Preprocessing option "{name}" is missing from PREPROCESSING_OPTIONS.',
        ) from exc


def _resolve_feature_option(
    *,
    name: str,
    method_path: str,
) -> FeatureOption:
    try:
        return FEATURE_OPTIONS.get_name(name=name)
    except KeyError:
        method = _import_attr(dotted_path=method_path)
        return FeatureOption(
            name=name,
            feature_channel_extraction_method=method,
        )


def _resolve_model_option(
    *,
    name: str,
    backend: Literal["torch", "sklearn"],
    target_kind: Literal["classification", "regression"],
    output_size: int | None,
    builder_path: str,
) -> ModelOption:
    try:
        base_option = MODEL_OPTIONS.get_name(name=name)
        builder = base_option.model_builder
    except KeyError:
        builder = _import_attr(dotted_path=builder_path)
    return ModelOption(
        name=name,
        model_builder=builder,
        target_kind=target_kind,
        backend=backend,
        output_size=output_size,
    )


def _resolve_channel_pick(channel_pick_params: dict[str, Any]) -> ChannelPickOption:
    return ChannelPickOption(
        name=channel_pick_params["name"],
        channel_pick=list(channel_pick_params["channel_pick"]),
    )


def _resolve_segmentation(segmentation_params: dict[str, Any]) -> SegmentationOption:
    return SegmentationOption(
        time_window=float(segmentation_params["time_window"]),
        time_step=float(segmentation_params["time_step"]),
    )


def _build_pipeline_context(run_dir: Path) -> PipelineContext:
    params = _load_json(path=run_dir / "params.json")
    training_option_params = params["training_option"]
    data_option_params = training_option_params["training_data_option"]
    build_dataset_params = data_option_params["build_dataset_option"]
    feature_extraction_params = data_option_params["feature_extraction_option"]

    artifacts = RunArtifacts(
        run_dir=run_dir,
        params=params,
        build_dataset_params=build_dataset_params,
        target_name=build_dataset_params["target"],
        target_kind_raw=str(build_dataset_params["target_kind"]),
        target_kind=_normalize_target_kind(str(build_dataset_params["target_kind"])),
        backend=training_option_params["training_method_option"]["backend"],
        class_labels=_class_labels_array(build_dataset_params=build_dataset_params),
    )

    preprocessing_option = _resolve_preprocessing_option(
        name=feature_extraction_params["preprocessing_option"]["name"],
    )
    channel_pick_option = _resolve_channel_pick(
        channel_pick_params=feature_extraction_params["channel_pick_option"],
    )
    feature_option_params = feature_extraction_params["feature_option"]
    feature_option = _resolve_feature_option(
        name=feature_option_params["name"],
        method_path=feature_option_params["feature_channel_extraction_method"],
    )
    segmentation_option = _resolve_segmentation(
        segmentation_params=feature_extraction_params["segmentation_option"],
    )
    feature_extraction_option = FeatureExtractionOption(
        preprocessing_option=preprocessing_option,
        feature_option=feature_option,
        channel_pick_option=channel_pick_option,
        segmentation_option=segmentation_option,
    )

    build_dataset_option = BuildDatasetOption(
        target=build_dataset_params["target"],
        random_seed=int(build_dataset_params["random_seed"]),
        use_size=float(build_dataset_params["use_size"]),
        test_size=float(build_dataset_params["test_size"]),
        target_kind=str(build_dataset_params["target_kind"]),
        feature_scaler=str(build_dataset_params["feature_scaler"]),
    )
    build_dataset_option.class_labels = (
        list(artifacts.class_labels) if artifacts.class_labels is not None else None
    )
    build_dataset_option.class_labels_expected = build_dataset_params.get(
        "class_labels_expected",
    )

    model_option_params = params["model_option"]
    model_option = _resolve_model_option(
        name=model_option_params["name"],
        backend=model_option_params["backend"],
        target_kind=artifacts.target_kind,
        output_size=model_option_params["output_size"],
        builder_path=model_option_params["model_builder"],
    )
    if artifacts.backend == "torch":
        if artifacts.target_kind == "classification":
            if model_option.output_size is None:
                raise ValueError("Torch classification models require output_size.")
            if (
                artifacts.class_labels is not None
                and len(artifacts.class_labels) != model_option.output_size
            ):
                raise ValueError(
                    "Torch model output_size does not match the class label count.",
                )
        elif model_option.output_size not in (None, 1):
            raise ValueError("Torch regression models must use output_size=1.")

    return PipelineContext(
        preprocessing_option=preprocessing_option,
        feature_extraction_option=feature_extraction_option,
        build_dataset_option=build_dataset_option,
        model_option=model_option,
        artifacts=artifacts,
    )


class _FeatureScaler:
    """Fit/transform helper that mirrors training-time scaling."""

    def __init__(self, build_dataset_option: BuildDatasetOption) -> None:
        self._build_dataset_option = build_dataset_option
        self._scaler: StandardScaler | MinMaxScaler | None = None

    def fit(self, feature_extraction_option: FeatureExtractionOption) -> None:
        scaler_name = self._build_dataset_option.feature_scaler
        if scaler_name == "none":
            return
        frames = [
            joblib.load(filename=path)
            for path in feature_extraction_option.get_file_paths()
        ]
        frame = cast(
            "pd.DataFrame",
            pd.concat(objs=frames, axis=0, ignore_index=True),
        )
        arrays = [
            np.asarray(a=feature, dtype=np.float32)
            for feature in frame["data"].tolist()
        ]
        flattened = np.stack(
            arrays=[array.reshape(-1) for array in arrays],
            axis=0,
        )
        if scaler_name == "standard":
            self._scaler = StandardScaler()
        elif scaler_name == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported feature_scaler: {scaler_name}")
        self._scaler.fit(flattened)

    def transform(self, feature: np.ndarray) -> np.ndarray:
        if self._build_dataset_option.feature_scaler == "none":
            return np.asarray(a=feature, dtype=np.float32)
        if self._scaler is None:
            raise RuntimeError("Feature scaler has not been fit.")
        flattened = feature.reshape(1, -1)
        scaled = self._scaler.transform(flattened)
        return scaled.reshape(feature.shape)


def _load_model_for_pipeline(pipeline: PipelineContext) -> BaseEstimator | torch.nn.Module:
    artifact = (
        pipeline.artifacts.run_dir / "best_model.pt"
        if pipeline.artifacts.backend == "torch"
        else pipeline.artifacts.run_dir / "best_model.joblib"
    )
    if pipeline.artifacts.backend == "torch":
        model = pipeline.model_option.build_model()
        state_dict = torch.load(f=artifact, map_location="cpu")
        torch_model = cast("torch.nn.Module", model)
        torch_model.load_state_dict(state_dict)
        torch_model.eval()
        return torch_model
    estimator = joblib.load(filename=artifact)
    if not isinstance(estimator, BaseEstimator):
        raise RuntimeError("Loaded sklearn artifact is not a BaseEstimator.")
    return estimator


def _predict_with_torch(
    *,
    pipeline: PipelineContext,
    model: torch.nn.Module,
    feature_vector: np.ndarray,
) -> tuple[float, list[float] | None]:
    input_tensor = torch.tensor(
        data=feature_vector.reshape(1, -1),
        dtype=torch.float32,
    )
    formatted = _ensure_conv1d_shape(batch=input_tensor)
    with torch.no_grad():
        outputs = model(formatted)
    if pipeline.artifacts.target_kind == "classification":
        probabilities = torch.softmax(input=outputs, dim=1).squeeze()
        pred_index = int(torch.argmax(probabilities, dim=0).item())
        class_labels = pipeline.artifacts.class_labels
        prediction = (
            float(class_labels[pred_index])
            if class_labels is not None
            else float(pred_index)
        )
        return prediction, probabilities.cpu().tolist()
    prediction_value = float(
        outputs.detach().cpu().reshape(-1)[0],
    )
    return prediction_value, None


def _predict_with_sklearn(
    *,
    pipeline: PipelineContext,
    model: BaseEstimator,
    feature_vector: np.ndarray,
) -> tuple[float, list[float] | None]:
    if pipeline.artifacts.target_kind == "classification":
        proba_method = getattr(model, "predict_proba", None)
        proba: np.ndarray | None = None
        if callable(proba_method):
            proba = proba_method(X=feature_vector.reshape(1, -1))
        pred_raw = model.predict(X=feature_vector.reshape(1, -1))
        pred_index = int(np.asarray(a=pred_raw).reshape(-1)[0])
        class_labels = pipeline.artifacts.class_labels
        prediction = (
            float(class_labels[pred_index])
            if class_labels is not None
            else float(pred_index)
        )
        probabilities = proba.reshape(-1).tolist() if proba is not None else None
        return prediction, probabilities
    pred_value = model.predict(X=feature_vector.reshape(1, -1))
    return float(np.asarray(a=pred_value).reshape(-1)[0]), None


class _TrialStream:
    """Yield per-segment features for trials in a looping stream."""

    def __init__(
        self,
        *,
        feature_extraction_option: FeatureExtractionOption,
        target_names: list[str],
        start_index: int | None,
    ) -> None:
        self._feature_extraction_option = feature_extraction_option
        self._target_names = target_names
        self._start_index = start_index

    def __iter__(self) -> Iterator[SegmentPayload]:
        trial_dir = self._feature_extraction_option.preprocessing_option.get_trial_path()
        trial_paths = sorted(trial_dir.glob("*.joblib"))
        if not trial_paths:
            raise FileNotFoundError(
                f'No trial data found at "{trial_dir}" for preprocessing option '
                f'{{{self._feature_extraction_option.preprocessing_option.name}}}.',
            )
        start = (
            self._start_index
            if self._start_index is not None
            else random.randrange(len(trial_paths))
        )
        for offset in itertools.count():
            idx = (start + offset) % len(trial_paths)
            trial_path = trial_paths[idx]
            trial_df = joblib.load(filename=trial_path)
            row = trial_df.iloc[0]
            subject_id = int(row["subject"])
            feature_df, _, _ = _extract_feature(
                trial_df=trial_df,
                feature_extraction_option=self._feature_extraction_option,
            )
            for segment_index, segment_row in enumerate(
                feature_df.itertuples(index=False),
            ):
                segment_array = np.asarray(a=segment_row.data, dtype=np.float32)
                start_seconds = BASELINE_SEC + (
                    segment_index
                    * self._feature_extraction_option.segmentation_option.time_step
                )
                targets: dict[str, float | None] = {}
                for target in self._target_names:
                    target_raw = getattr(segment_row, target, None)
                    targets[target] = float(target_raw) if target_raw is not None else None
                yield SegmentPayload(
                    feature=segment_array,
                    subject=subject_id,
                    trial=int(row["trial"]),
                    segment_index=segment_index,
                    segment_start_seconds=start_seconds,
                    targets=targets,
                )


class RealtimePredictor:
    """Bridge that streams raw segments through the trained model."""

    def __init__(self, pipeline: PipelineContext, request: StreamingRequest) -> None:
        self._pipeline = pipeline
        self._request = request
        self._scaler = _FeatureScaler(
            build_dataset_option=pipeline.build_dataset_option,
        )
        self._model = _load_model_for_pipeline(pipeline=pipeline)
        self._scaler.fit(feature_extraction_option=pipeline.feature_extraction_option)

    def _predict_torch(
        self,
        *,
        feature_vector: np.ndarray,
    ) -> tuple[float, list[float] | None]:
        torch_model = cast("torch.nn.Module", self._model)
        return _predict_with_torch(
            pipeline=self._pipeline,
            model=torch_model,
            feature_vector=feature_vector,
        )

    def _predict_sklearn(
        self,
        *,
        feature_vector: np.ndarray,
    ) -> tuple[float, list[float] | None]:
        estimator = cast("BaseEstimator", self._model)
        return _predict_with_sklearn(
            pipeline=self._pipeline,
            model=estimator,
            feature_vector=feature_vector,
        )

    def stream_predictions(self) -> Iterator[PredictionEvent]:
        stream = _TrialStream(
            feature_extraction_option=self._pipeline.feature_extraction_option,
            target_names=[self._pipeline.artifacts.target_name],
            start_index=self._request.start_trial_index,
        )
        for segment in stream:
            scaled = self._scaler.transform(feature=segment.feature)
            if self._pipeline.artifacts.backend == "torch":
                prediction, probabilities = self._predict_torch(
                    feature_vector=scaled.reshape(-1),
                )
            else:
                prediction, probabilities = self._predict_sklearn(
                    feature_vector=scaled.reshape(-1),
                )
            yield PredictionEvent(
                prediction=prediction,
                subject=segment.subject,
                trial=segment.trial,
                segment_index=segment.segment_index,
                segment_start_seconds=segment.segment_start_seconds,
                target_value=segment.targets.get(
                    self._pipeline.artifacts.target_name,
                ),
                probabilities=probabilities,
            )
            if self._request.segment_pause_seconds > 0:
                time.sleep(self._request.segment_pause_seconds)


class CombinedRealtimePredictor:
    """Emit paired arousal/valence predictions per incoming segment."""

    def __init__(self, pipelines: CombinedPipelines, request: StreamingRequest) -> None:
        self._pipelines = pipelines
        self._request = request

        self._arousal_scaler = _FeatureScaler(
            build_dataset_option=pipelines.arousal.build_dataset_option,
        )
        self._valence_scaler = _FeatureScaler(
            build_dataset_option=pipelines.valence.build_dataset_option,
        )
        self._arousal_model = _load_model_for_pipeline(pipeline=pipelines.arousal)
        self._valence_model = _load_model_for_pipeline(pipeline=pipelines.valence)

        self._arousal_scaler.fit(
            feature_extraction_option=pipelines.arousal.feature_extraction_option,
        )
        self._valence_scaler.fit(
            feature_extraction_option=pipelines.valence.feature_extraction_option,
        )

    def stream_predictions(self) -> Iterator[CombinedPredictionEvent]:
        stream = _TrialStream(
            feature_extraction_option=self._pipelines.arousal.feature_extraction_option,
            target_names=[
                self._pipelines.arousal.artifacts.target_name,
                self._pipelines.valence.artifacts.target_name,
            ],
            start_index=self._request.start_trial_index,
        )

        for segment in stream:
            scaled_arousal = self._arousal_scaler.transform(feature=segment.feature)
            scaled_valence = self._valence_scaler.transform(feature=segment.feature)

            if self._pipelines.arousal.artifacts.backend == "torch":
                arousal_pred, arousal_probs = _predict_with_torch(
                    pipeline=self._pipelines.arousal,
                    model=cast("torch.nn.Module", self._arousal_model),
                    feature_vector=scaled_arousal.reshape(-1),
                )
            else:
                arousal_pred, arousal_probs = _predict_with_sklearn(
                    pipeline=self._pipelines.arousal,
                    model=cast("BaseEstimator", self._arousal_model),
                    feature_vector=scaled_arousal.reshape(-1),
                )

            if self._pipelines.valence.artifacts.backend == "torch":
                valence_pred, valence_probs = _predict_with_torch(
                    pipeline=self._pipelines.valence,
                    model=cast("torch.nn.Module", self._valence_model),
                    feature_vector=scaled_valence.reshape(-1),
                )
            else:
                valence_pred, valence_probs = _predict_with_sklearn(
                    pipeline=self._pipelines.valence,
                    model=cast("BaseEstimator", self._valence_model),
                    feature_vector=scaled_valence.reshape(-1),
                )

            yield CombinedPredictionEvent(
                subject=segment.subject,
                trial=segment.trial,
                segment_index=segment.segment_index,
                segment_start_seconds=segment.segment_start_seconds,
                arousal_prediction=arousal_pred,
                valence_prediction=valence_pred,
                arousal_target=segment.targets.get(
                    self._pipelines.arousal.artifacts.target_name,
                ),
                valence_target=segment.targets.get(
                    self._pipelines.valence.artifacts.target_name,
                ),
                arousal_probabilities=arousal_probs,
                valence_probabilities=valence_probs,
            )

            if self._request.segment_pause_seconds > 0:
                time.sleep(self._request.segment_pause_seconds)


def build_mock_pipeline(run_dir: Path) -> PipelineContext:
    """
    Rebuild preprocessing/feature/model options from a stored run directory.

    Parameters
    ----------
    run_dir:
        Path to a run directory under ``results/`` that contains ``params.json``
        and model artifacts.
    """
    return _build_pipeline_context(run_dir=run_dir)


def build_dual_pipeline(
    *,
    arousal_run_dir: Path,
    valence_run_dir: Path,
) -> CombinedPipelines:
    """
    Construct paired pipelines for arousal/valence using shared features.

    Raises
    ------
    ValueError
        If the two runs do not share the same feature extraction configuration.
    """
    arousal = build_mock_pipeline(run_dir=arousal_run_dir)
    valence = build_mock_pipeline(run_dir=valence_run_dir)
    return CombinedPipelines(arousal=arousal, valence=valence)


def stream_realtime_predictions(
    pipeline: PipelineContext,
    request: StreamingRequest,
) -> Iterator[PredictionEvent]:
    """
    High-level helper that returns an iterator over live predictions.

    Parameters
    ----------
    pipeline:
        Pipeline reconstructed via ``build_mock_pipeline``.
    request:
        Streaming request controlling which subjects to stream and the pacing
        between segments.
    """
    predictor = RealtimePredictor(pipeline=pipeline, request=request)
    return predictor.stream_predictions()


def stream_dual_realtime_predictions(
    *,
    pipelines: CombinedPipelines,
    request: StreamingRequest,
) -> Iterator[CombinedPredictionEvent]:
    """
    Emit both arousal and valence predictions for each streamed segment.

    Parameters
    ----------
    pipelines:
        Paired arousal/valence pipelines created by ``build_dual_pipeline``.
    request:
        Streaming controls (subjects, max trials, pacing).
    """
    predictor = CombinedRealtimePredictor(pipelines=pipelines, request=request)
    return predictor.stream_predictions()
