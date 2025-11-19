"""Development entry point for running preprocessing + extraction locally."""

from __future__ import annotations

from itertools import product
from typing import Any

from feature_extractor import (
    CHANNEL_PICK_OPTIONS,
    FEATURE_OPTIONS,
    SEGMENTATION_OPTIONS,
    run_feature_extractor,
)
from feature_extractor.types import FeatureExtractionOption
from model_trainer import run_model_trainer
from model_trainer.options import MODEL_OPTIONS, TRAINING_METHOD_OPTIONS
from model_trainer.types import (
    ModelTrainingOption,
    TrainingDataOption,
    TrainingOption,
)
from preprocessor import run_preprocessor
from preprocessor.options import PREPROCESSING_OPTIONS

_EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    {
        "target": "valence",
        "target_kind": "regression",
        "feature_scaler": "minmax",
        "training_method_name": "adam_regression",
        "model_name": "cnn1d_n1_regression",
        "class_labels_expected": None,
    },
    {
        "target": "valence",
        "target_kind": "classification",
        "feature_scaler": "minmax",
        "training_method_name": "adam_classification",
        "model_name": "cnn1d_n1_classification",
        "class_labels_expected": [float(index) for index in range(1, 10)],
    },
]


def _run_extraction_examples() -> None:
    """Execute a small subset of preprocessing + feature extraction variants."""
    run_preprocessor(preprocessing_option=PREPROCESSING_OPTIONS.get_name(name="clean"))
    # run_preprocessor(
    #     preprocessing_option=PREPROCESSING_OPTIONS.get_name(name="ica_clean"),
    # )

    feop_keys = [
        "preprocessing_option",
        "feature_option",
        "channel_pick_option",
        "segmentation_option",
    ]
    feop_values = [
        PREPROCESSING_OPTIONS.get_names(names=["clean"]),
        FEATURE_OPTIONS.get_names(["psd"]),
        CHANNEL_PICK_OPTIONS.get_names(names=["standard_32"]),
        SEGMENTATION_OPTIONS,
    ]
    feop_combos = [
        dict(zip(feop_keys, feop_combo, strict=True))
        for feop_combo in product(*feop_values)
    ]

    for combo in feop_combos:
        fe_option = FeatureExtractionOption(**combo)
        run_feature_extractor(feature_extraction_option=fe_option)

        for experiment in _EXPERIMENT_CONFIGS:
            training_data_option = TrainingDataOption(
                feature_extraction_option=fe_option,
                target=experiment["target"],
                random_seed=42,
                use_size=0.3,
                test_size=0.2,
                target_kind=experiment["target_kind"],
                feature_scaler=experiment["feature_scaler"],
                class_labels_expected=experiment["class_labels_expected"],
            )
            training_method_option = TRAINING_METHOD_OPTIONS.get_name(
                name=experiment["training_method_name"],
            )
            training_option = TrainingOption(
                training_data_option=training_data_option,
                training_method_option=training_method_option,
            )
            model_option = MODEL_OPTIONS.get_name(name=experiment["model_name"])
            model_training_option = ModelTrainingOption(
                model_option=model_option,
                training_option=training_option,
            )
            run_model_trainer(model_training_option=model_training_option)


if __name__ == "__main__":
    _run_extraction_examples()
