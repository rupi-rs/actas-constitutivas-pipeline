"""Domain exception hierarchy for the deed automation pipeline.

Every error raised intentionally by pipeline components derives from
:class:`PipelineError`, allowing the CLI to catch a single base type and
render a sober ``[ERROR]`` line while preserving the specific failure class.
"""

from __future__ import annotations

from typing import Any, Optional


class PipelineError(Exception):
    """Base class for all intentional pipeline failures.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    details : dict of str to Any, optional
        Structured context attached to the audit log record.

    Attributes
    ----------
    message : str
        Human-readable description of the failure.
    details : dict of str to Any
        Structured context (never ``None``).
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initialise the exception.

        Parameters
        ----------
        message : str
            Human-readable description of the failure.
        details : dict of str to Any, optional
            Structured context attached to the audit log record.
        """
        super().__init__(message)
        self.message: str = message
        self.details: dict[str, Any] = dict(details) if details else {}


class ConfigurationError(PipelineError):
    """Raised when settings or environment configuration are invalid."""


class IngestionError(PipelineError):
    """Raised when a document cannot be read, hashed or opened as a PDF."""


class PDFInspectionError(PipelineError):
    """Raised when page-level native text inspection or rendering fails."""


class OCRError(PipelineError):
    """Raised when an OCR provider cannot produce a result for a page."""


class VerifierError(PipelineError):
    """Raised when the predicate verifier (Escalation Tier 2) fails."""


class VisionReadError(PipelineError):
    """Raised when the VLM re-read of a low-confidence region fails or is refused."""


class ExtractionError(PipelineError):
    """Raised when structured LLM extraction fails or returns invalid data."""


class CacheMissError(PipelineError):
    """Raised when a deterministic provider finds no pre-verified cache entry."""


class ResolutionError(PipelineError):
    """Raised when temporal entity resolution cannot be completed."""


class ValidationError(PipelineError):
    """Raised when deterministic validation encounters an unrecoverable state.

    Notes
    -----
    Distinct from :class:`pydantic.ValidationError`; import with care.
    """


class StorageError(PipelineError):
    """Raised when cache or output persistence fails."""


class UnknownDocumentTypeError(PipelineError):
    """Raised when a document type is not registered in the registry."""


class EvaluationError(PipelineError):
    """Raised when outputs cannot be compared against ground truth."""
