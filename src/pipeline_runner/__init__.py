"""Public API for launching complete EEG processing pipelines."""

from .core import (
    TrainingExperiment as TrainingExperiment,
)
from .core import (
    build_feature_extraction_options as build_feature_extraction_options,
)
from .core import (
    run_pipeline as run_pipeline,
)

__all__ = ["TrainingExperiment", "build_feature_extraction_options", "run_pipeline"]
