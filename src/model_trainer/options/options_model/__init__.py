"""Registered neural-network architectures for the trainer."""

from __future__ import annotations

from config.option_utils import OptionList
from model_trainer.types import ModelOption

from .option_bihdm import BiHDM
from .option_cfcnn import CFCNN
from .option_cnn1d_n1 import CNN1D_N1
from .option_combined_mlp import CombinedWaveletMLP
from .option_kmeans_mlp import KMeansWaveletMLP
from .option_mlp import _mlp_classification
from .option_sklearn_baseline import (
    _sklearn_elasticnet_regression,
    _sklearn_gb_regression,
    _sklearn_knn_classifier,
    _sklearn_linear_regression,
    _sklearn_linear_svc,
    _sklearn_logreg,
    _sklearn_qda_classifier,
    _sklearn_rf_classification,
    _sklearn_rf_regression,
    _sklearn_ridge_regression,
    _sklearn_sgd_classifier,
    _sklearn_svc_rbf,
    _sklearn_svr_rbf,
)


def _build_cnn1d_n1(*, output_size: int) -> CNN1D_N1:
    return CNN1D_N1(output_size=output_size)


def _build_cfcnn(*, output_size: int) -> CFCNN:
    return CFCNN(output_size=output_size)


def _build_combined_wavelet_mlp(*, output_size: int) -> CombinedWaveletMLP:
    return CombinedWaveletMLP(output_size=output_size)


def _build_kmeans_wavelet_mlp(*, output_size: int) -> KMeansWaveletMLP:
    return KMeansWaveletMLP(output_size=output_size)


def _build_bihdm(*, output_size: int) -> BiHDM:
    return BiHDM(output_size=output_size)


__all__ = ["MODEL_OPTIONS"]

_cnn1d_n1_regression = ModelOption(
    name="cnn1d_n1_regression",
    model_builder=_build_cnn1d_n1,
    output_size=1,
    backend="torch",
    target_kind="regression",
)

_cnn1d_n1_classification = ModelOption(
    name="cnn1d_n1_classification",
    model_builder=_build_cnn1d_n1,
    output_size=9,
    backend="torch",
    target_kind="classification",
)

_cfcnn_classification = ModelOption(
    name="cfcnn_classification",
    model_builder=_build_cfcnn,
    output_size=9,
    backend="torch",
    target_kind="classification",
)

_combined_wavelet_mlp_classification = ModelOption(
    name="combined_wavelet_mlp_classification",
    model_builder=_build_combined_wavelet_mlp,
    output_size=9,
    backend="torch",
    target_kind="classification",
)

_kmeans_wavelet_mlp_classification = ModelOption(
    name="kmeans_wavelet_mlp_classification",
    model_builder=_build_kmeans_wavelet_mlp,
    output_size=9,
    backend="torch",
    target_kind="classification",
)

_bihdm_classification = ModelOption(
    name="bihdm_classification",
    model_builder=_build_bihdm,
    output_size=9,
    backend="torch",
    target_kind="classification",
)

MODEL_OPTIONS: OptionList[ModelOption] = OptionList(
    options=[
        _cnn1d_n1_regression,
        _cnn1d_n1_classification,
        _cfcnn_classification,
        _combined_wavelet_mlp_classification,
        _kmeans_wavelet_mlp_classification,
        _bihdm_classification,
        _mlp_classification,
        _sklearn_logreg,
        _sklearn_rf_regression,
        _sklearn_rf_classification,
        _sklearn_svc_rbf,
        _sklearn_linear_svc,
        _sklearn_svr_rbf,
        _sklearn_knn_classifier,
        _sklearn_qda_classifier,
        _sklearn_gb_regression,
        _sklearn_sgd_classifier,
        _sklearn_ridge_regression,
        _sklearn_linear_regression,
        _sklearn_elasticnet_regression,
    ],
)
