"""Registered neural-network architectures for the trainer."""

from __future__ import annotations

from config.option_utils import OptionList
from model_trainer.types import ModelOption

from .option_cnn1d_n1 import CNN1D_N1


def _build_cnn1d_n1(*, output_size: int) -> CNN1D_N1:
    return CNN1D_N1(output_size=output_size)


__all__ = ["MODEL_OPTIONS"]

_cnn1d_n1_regression = ModelOption(
    name="cnn1d_n1_regression",
    model_builder=_build_cnn1d_n1,
    output_size=1,
    target_kind="regression",
)

_cnn1d_n1_classification = ModelOption(
    name="cnn1d_n1_classification",
    model_builder=_build_cnn1d_n1,
    output_size=9,
    target_kind="classification",
)

MODEL_OPTIONS: OptionList[ModelOption] = OptionList(
    options=[_cnn1d_n1_regression, _cnn1d_n1_classification],
)
