"""Domain-specific exceptions with stable CLI exit categories."""

from typing import Any


class ProcessorError(Exception):
    """Base class for expected user-facing failures."""


class ConfigurationError(ProcessorError):
    """Invalid command configuration or unsupported input."""


class PlatformError(ProcessorError):
    """Runtime platform cannot execute the local MLX pipeline."""


class MediaError(ProcessorError):
    """Input media cannot be inspected or normalized."""


class ModelUnavailableError(ProcessorError):
    """A required model is absent from the local cache."""


class BackendError(ProcessorError):
    """An ASR backend failed to return a usable result."""

    def __init__(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.metadata = metadata or {}
        self.warnings = warnings


class SchemaError(ProcessorError):
    """A transcript JSON does not satisfy the canonical schema."""


class OutputExistsError(ProcessorError):
    """A destination exists and overwrite was not authorized."""


class TranscriptionFailed(ProcessorError):
    """Every allowed backend produced an objective hard failure."""

    def __init__(self, message: str, *, diagnostic_path=None) -> None:
        super().__init__(message)
        self.diagnostic_path = diagnostic_path
