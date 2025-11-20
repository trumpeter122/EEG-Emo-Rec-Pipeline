"""High-level orchestration helpers for running full EEG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal

from feature_extractor import run_feature_extractor
from feature_extractor.types import (
    ChannelPickOption,
    FeatureExtractionOption,
    FeatureOption,
    SegmentationOption,
)
from model_trainer import run_model_trainer
from model_trainer.types import (
    ModelOption,
    ModelTrainingOption,
    TrainingDataOption,
    TrainingMethodOption,
    TrainingOption,
)
from preprocessor import run_preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from config.option_utils import OptionList
    from preprocessor.types import PreprocessingOption

__all__ = ["TrainingExperiment", "build_feature_extraction_options", "run_pipeline"]


@dataclass(slots=True)
class TrainingExperiment:
    """
    Blueprint for a single model-training configuration.

    Each instance specifies the downstream label column, whether the task is
    a regression or classification problem, how segment features should be
    scaled, which optimizer/criterion bundle to use, and which registered
    model should be instantiated.  The dataset split settings and random seed
    allow contributors to describe reproducible training/evaluation pairs
    without touching the orchestration logic.
    """

    target: str
    target_kind: Literal["regression", "classification"]
    feature_scaler: Literal["none", "standard", "minmax"]
    training_method_name: str
    model_name: str
    random_seed: int
    use_size: float
    test_size: float
    class_labels_expected: Sequence[float] | None = None


def build_feature_extraction_options(
    *,
    preprocessing_options: Sequence[PreprocessingOption],
    feature_options: Sequence[FeatureOption],
    channel_pick_options: Sequence[ChannelPickOption],
    segmentation_options: Sequence[SegmentationOption],
) -> list[FeatureExtractionOption]:
    """
    Materialize every valid ``FeatureExtractionOption`` combination.

    The helper enforces that each option sequence contains at least one entry
    and then walks the Cartesian product of the four option families.  The
    result is a list of fully-qualified options ready to be passed into
    ``run_feature_extractor`` and later reused when constructing training
    datasets.
    """
    if not preprocessing_options:
        raise ValueError("preprocessing_options cannot be empty.")
    if not feature_options:
        raise ValueError("feature_options cannot be empty.")
    if not channel_pick_options:
        raise ValueError("channel_pick_options cannot be empty.")
    if not segmentation_options:
        raise ValueError("segmentation_options cannot be empty.")

    combos = []
    for (
        preprocessing_option,
        feature_option,
        channel_pick_option,
        segmentation_option,
    ) in product(
        preprocessing_options,
        feature_options,
        channel_pick_options,
        segmentation_options,
    ):
        combos.append(
            FeatureExtractionOption(
                preprocessing_option=preprocessing_option,
                feature_option=feature_option,
                channel_pick_option=channel_pick_option,
                segmentation_option=segmentation_option,
            ),
        )
    return combos


def run_pipeline(
    *,
    preprocessing_options: Sequence[PreprocessingOption],
    feature_options: Sequence[FeatureOption],
    channel_pick_options: Sequence[ChannelPickOption],
    segmentation_options: Sequence[SegmentationOption],
    experiments: Sequence[TrainingExperiment],
    model_options: OptionList[ModelOption],
    training_method_options: OptionList[TrainingMethodOption],
) -> None:
    """
    Execute preprocessing, feature extraction, and model training pipelines.

    The function is intentionally declarative: callers provide curated lists
    of preprocessing/feature/segmentation/channel options plus a collection
    of ``TrainingExperiment`` descriptors.  The runner first ensures each
    preprocessing option has generated the required intermediate files,
    exhaustively extracts features for every combination, and finally spins
    up the requested training runs, writing artifacts under the standard
    ``results/`` hierarchy.
    """
    if not experiments:
        raise ValueError("experiments cannot be empty.")

    for preprocessing_option in preprocessing_options:
        run_preprocessor(preprocessing_option=preprocessing_option)

    feature_extraction_options = build_feature_extraction_options(
        preprocessing_options=preprocessing_options,
        feature_options=feature_options,
        channel_pick_options=channel_pick_options,
        segmentation_options=segmentation_options,
    )

    for feature_extraction_option in feature_extraction_options:
        run_feature_extractor(feature_extraction_option=feature_extraction_option)
        for experiment in experiments:
            model_training_option = _build_model_training_option(
                feature_extraction_option=feature_extraction_option,
                experiment=experiment,
                model_options=model_options,
                training_method_options=training_method_options,
            )
            run_model_trainer(model_training_option=model_training_option)


def _build_model_training_option(
    *,
    feature_extraction_option: FeatureExtractionOption,
    experiment: TrainingExperiment,
    model_options: OptionList[ModelOption],
    training_method_options: OptionList[TrainingMethodOption],
) -> ModelTrainingOption:
    """
    Assemble a ``ModelTrainingOption`` from registry-driven components.

    The helper wires the current ``FeatureExtractionOption`` into a freshly
    minted ``TrainingDataOption`` (which handles frame loading, encoding, and
    scaling), attaches the named training method to create dataloaders, and
    selects the registered neural network architecture requested by the
    experiment.  The returned object fully describes the run for
    ``run_model_trainer`` as well as for downstream metadata serialization.
    """
    training_data_option = TrainingDataOption(
        feature_extraction_option=feature_extraction_option,
        target=experiment.target,
        random_seed=experiment.random_seed,
        use_size=experiment.use_size,
        test_size=experiment.test_size,
        target_kind=experiment.target_kind,
        feature_scaler=experiment.feature_scaler,
        class_labels_expected=experiment.class_labels_expected,
    )
    training_method_option = training_method_options.get_name(
        name=experiment.training_method_name,
    )
    training_option = TrainingOption(
        training_data_option=training_data_option,
        training_method_option=training_method_option,
    )
    model_option = model_options.get_name(name=experiment.model_name)
    return ModelTrainingOption(
        model_option=model_option,
        training_option=training_option,
    )
