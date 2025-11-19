"""Shared option utilities and protocols."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial as _partial
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

if TYPE_CHECKING:
    import mne  # type: ignore[import-untyped]
    import numpy as np
    from torch import nn
    from torch.optim import Optimizer

__all__ = [
    "CriterionBuilder",
    "FeatureChannelExtractionMethod",
    "ModelBuilder",
    "OptionList",
    "OptimizerBuilder",
    "PreprocessingMethod",
    "_callable_path",
]


class _NamedOption(Protocol):
    """Protocol describing option objects with a ``name`` attribute."""

    name: str


_OptionType = TypeVar("_OptionType", bound=_NamedOption)


def _callable_path(callable_obj: Callable[..., Any]) -> str:
    """Return a dotted path for serializing callable references."""
    if isinstance(callable_obj, _partial):
        func_path = _callable_path(callable_obj.func)
        return f"functools.partial({func_path})"

    module = getattr(callable_obj, "__module__", None)
    qualname = getattr(
        callable_obj,
        "__qualname__",
        getattr(callable_obj, "__name__", None),
    )
    if module and qualname:
        return f"{module}.{qualname}"

    return repr(callable_obj)


class PreprocessingMethod(Protocol):
    """Callable signature for preprocessing functions."""

    def __call__(self, raw: mne.io.BaseRaw, subject_id: int) -> np.ndarray: ...


class FeatureChannelExtractionMethod(Protocol):
    """Callable signature for feature extraction per segment."""

    def __call__(
        self,
        trial_data: np.ndarray,
        channel_pick: list[str],
    ) -> np.ndarray: ...


class ModelBuilder(Protocol):
    """Callable protocol for constructing neural network models."""

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module: ...


class OptimizerBuilder(Protocol):
    """Callable protocol for instantiating optimizers."""

    def __call__(self, *, model: nn.Module) -> Optimizer: ...


class CriterionBuilder(Protocol):
    """Callable protocol for generating loss functions."""

    def __call__(self) -> nn.Module: ...


class OptionList(Sequence[_OptionType]):
    """Ordered registry of option objects with lookup helpers."""

    def __init__(self, options: Iterable[_OptionType]):
        self._options = list(options)

    @overload
    def __getitem__(self, index: int) -> _OptionType: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_OptionType]: ...

    def __getitem__(self, index: int | slice) -> _OptionType | Sequence[_OptionType]:
        return self._options[index]

    def __len__(self) -> int:
        return len(self._options)

    def __iter__(self) -> Iterator[_OptionType]:
        return iter(self._options)

    def get_name(self, name: str) -> _OptionType:
        """
        Retrieve a single option by name.

        Raises
        ------
        KeyError
            If the requested name does not exist in the collection.
        """
        for option in self._options:
            if option.name == name:
                return option

        raise KeyError(f'Name "{name}" does not exist.')

    def get_names(self, names: Sequence[str]) -> list[_OptionType]:
        """Return options that match ``names`` in order."""
        return [self.get_name(name=name) for name in names]
