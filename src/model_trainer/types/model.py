"""Neural-network option definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from config.option_utils import ModelBuilder, _callable_path

if TYPE_CHECKING:
    from torch import nn

__all__ = ["ModelOption"]


@dataclass(slots=True)
class ModelOption:
    """
    Descriptor for constructing a specific neural network architecture.

    - Options store a human-readable name, the callable ``model_builder`` that
      produces initialized ``nn.Module`` instances, the required ``output_size``,
      and the supported ``target_kind``.
    - Serialization helpers expose this data so metrics/artifacts can be traced
      back to the exact model configuration.
    """

    name: str
    model_builder: ModelBuilder
    output_size: int
    target_kind: Literal["regression", "classification"]

    def build_model(self) -> nn.Module:
        """Instantiate the configured model."""
        return self.model_builder(output_size=self.output_size)

    def to_params(self) -> dict[str, Any]:
        """Serialize the model descriptor."""
        return {
            "name": self.name,
            "model_builder": _callable_path(self.model_builder),
            "output_size": self.output_size,
            "target_kind": self.target_kind,
        }
