"""Development entry point for running preprocessing + extraction locally."""

from __future__ import annotations

from feature_extractor.options import (
    CHANNEL_PICK_OPTIONS,
    FEATURE_OPTIONS,
    SEGMENTATION_OPTIONS,
)
from model_trainer.options import (
    BUILD_DATASET_OPTIONS,
    MODEL_OPTIONS,
    TRAINING_METHOD_OPTIONS,
)
from pipeline_runner import run_pipeline
from preprocessor.options import PREPROCESSING_OPTIONS

print(
    "\n".join(
        [
            "PREPROCESSING_OPTIONS:",
            f"{PREPROCESSING_OPTIONS}",
            "",
            "CHANNEL_PICK_OPTIONS:",
            f"{CHANNEL_PICK_OPTIONS}",
            "",
            "FEATURE_OPTIONS:",
            f"{FEATURE_OPTIONS}",
            "",
            "SEGMENTATION_OPTIONS:",
            f"{SEGMENTATION_OPTIONS}",
            "",
            "BUILD_DATASET_OPTIONS:",
            f"{BUILD_DATASET_OPTIONS}",
            "",
            "MODEL_OPTIONS:",
            f"{MODEL_OPTIONS}",
            "",
            "TRAINING_METHOD_OPTIONS:",
            f"{TRAINING_METHOD_OPTIONS}",
            "",
        ]
    )
)

# Run for some combinations
# run_pipeline(
#     preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean", "unclean"]),
#     channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(
#         [
#             "minimal_frontal_parietal",
#             "balanced_classic_6",
#             "optimized_gold_standard_8",
#             "standard_32",
#         ]
#     ),
#     feature_options=FEATURE_OPTIONS.get_names(["psd", "deasm", "dasm", "de"]),
#     segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
#     model_options=MODEL_OPTIONS.get_names(
#         # ["cnn1d_n1_regression", "cnn1d_n1_classification"]
#         ["sgd_classifier_sklearn"]
#     ),
#     build_dataset_options=BUILD_DATASET_OPTIONS,
#     training_method_options=TRAINING_METHOD_OPTIONS.get_names(
#         [
#             "adam_regression",
#             "adam_classification",
#             "sklearn_default_classification",
#             "sklearn_default_regression",
#         ]
#     ),
# )

# Add more here
# run_pipeline()


def run_research_paper_01(extend: bool = False) -> None:
    """Replicate Research Paper 01 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["ica_clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(
                ["frontal_prefrontal_4"]
            ),
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s2.00"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.20+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.20+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s2.00"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_02(extend: bool = False) -> None:
    """Replicate Research Paper 02 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["unclean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["emotiv_frontal_3"]),
            feature_options=FEATURE_OPTIONS.get_names(["higuchi_fd_frontal"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s4.00"]),
            model_options=MODEL_OPTIONS.get_names(["linear_svc_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["higuchi_fd_frontal"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s4.00"]),
            model_options=MODEL_OPTIONS.get_names(["linear_svc_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_03(extend: bool = False) -> None:
    """Replicate Research Paper 03 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )


def run_research_paper_04(extend: bool = False) -> None:
    """Replicate Research Paper 04 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["emotiv_frontal_4"]),
            feature_options=FEATURE_OPTIONS.get_names(["hoc_stat_fd_frontal4"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s1.00"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["hoc_stat_fd_frontal4"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s1.00"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_05(extend: bool = False) -> None:
    """Replicate Research Paper 05 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["svc_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_06(extend: bool = False) -> None:
    """Replicate Research Paper 06 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["ridge_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.20+seed42+regression+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.20+seed42+regression+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["ridge_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )


def run_research_paper_07(extend: bool = False) -> None:
    """Replicate Research Paper 07 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )


def run_research_paper_08(extend: bool = False) -> None:
    """Replicate Research Paper 08 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["linear_svc_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["linear_svc_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_09(extend: bool = False) -> None:
    """Replicate Research Paper 09 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["unclean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(
                ["posterior_parietal_14"]
            ),
            feature_options=FEATURE_OPTIONS.get_names(["raw_waveform"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w1.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["linear_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.20+seed42+regression+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.20+seed42+regression+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["raw_waveform"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w1.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["linear_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )


def run_research_paper_10(extend: bool = False) -> None:
    """Replicate Research Paper 10 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s4.00"]),
            model_options=MODEL_OPTIONS.get_names(["mlp_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["wavelet_energy_entropy_stats"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s4.00"]),
            model_options=MODEL_OPTIONS.get_names(["mlp_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )


def run_research_paper_11(extend: bool = False) -> None:
    """Replicate Research Paper 11 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(
                ["minimal_temporal_augmented"]
            ),
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w1.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["svr_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.20+seed42+regression+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.20+seed42+regression+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["psd"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w1.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["svr_rbf_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )


def run_research_paper_12(extend: bool = False) -> None:
    """Replicate Research Paper 12 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["ica_clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["deasm"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["logreg_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["deasm"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["logreg_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_classification"]
            ),
        )


def run_research_paper_13(extend: bool = False) -> None:
    """Replicate Research Paper 13 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["deasm"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.30+seed42+classification+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.30+seed42+classification+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["deasm"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["cnn1d_n1_classification"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["adam_classification"]
            ),
        )


def run_research_paper_14(extend: bool = False) -> None:
    """Replicate Research Paper 14 configuration."""
    if not extend:
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
            channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["standard_32"]),
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["elasticnet_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                ["valence+use1.00+test0.20+seed42+regression+standard"]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )
    else:
        base_ds = "valence+use1.00+test0.20+seed42+regression+standard"
        run_pipeline(
            preprocessing_options=PREPROCESSING_OPTIONS.get_names(
                ["unclean", "clean", "ica_clean"]
            ),
            channel_pick_options=CHANNEL_PICK_OPTIONS,
            feature_options=FEATURE_OPTIONS.get_names(["de"]),
            segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
            model_options=MODEL_OPTIONS.get_names(["elasticnet_regression_sklearn"]),
            build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
                [base_ds, base_ds.replace("valence", "arousal", 1)]
            ),
            training_method_options=TRAINING_METHOD_OPTIONS.get_names(
                ["sklearn_default_regression"]
            ),
        )


# run_research_paper_01() # SVC_RBF: TLDR
run_research_paper_02()
run_research_paper_03()
# run_research_paper_04() # SVC_RBF: TLDR
# run_research_paper_05() # SVC_RBF: TLDR
run_research_paper_06()
run_research_paper_08()
run_research_paper_09()
run_research_paper_10()
# run_research_paper_11() # SVC_RBF: TLDR
run_research_paper_12()
run_research_paper_13()
run_research_paper_14()
# run_research_paper_07() # CNN: TL

# run_research_paper_01(extend=True) # SVC_RBF: TLDR
run_research_paper_02(extend=True)
run_research_paper_03(extend=True)
# run_research_paper_04(extend=True) # SVC_RBF: TLDR
# run_research_paper_05(extend=True) # SVC_RBF: TLDR
run_research_paper_06(extend=True)
run_research_paper_08(extend=True)
run_research_paper_09(extend=True)
run_research_paper_10(extend=True)
# run_research_paper_11(extend=True) # SVC_RBF: TLDR
run_research_paper_12(extend=True)
run_research_paper_13(extend=True)
run_research_paper_14(extend=True)
# run_research_paper_07(extend=True) # CNN: TL
