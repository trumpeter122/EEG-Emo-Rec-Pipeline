"""Custom exception types for configuration validation."""


class MissingAsymmetryPairsError(ValueError):
    """Raised when an asymmetry feature is requested without valid channel pairs."""
