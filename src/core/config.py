"""Centralised, immutable runtime configuration for the deed pipeline.

The settings object resolves filesystem locations relative to the project
root and exposes every decision threshold used by the inspector, OCR
escalation tiers, resolver, validators and evaluator. Values are read from
environment variables once and cached.

Notes
-----
Recognised environment variables:

- ``ACTAS_PROJECT_ROOT``: overrides the project root directory.
- ``ACTAS_AUDIT_LOG_PATH``: overrides the JSON Lines audit trail location.
- ``OPENAI_API_KEY``: enables live LLM / verifier execution when non-empty.
- ``OPENAI_EXTRACTION_MODEL``: model for live structured extraction through the
  Responses API; falls back to ``OPENAI_MODEL`` and then ``gpt-6-luna``.
- ``OPENAI_MODEL``: legacy alias used when ``OPENAI_EXTRACTION_MODEL`` is unset.
- ``OPENAI_EXTRACTION_TEMPERATURE``: optional sampling temperature for live
  extraction; when unset the parameter is omitted from the request.
- ``OPENAI_DECISIONS_MODEL``: model for the Tier 2 Decisions API verifier
  (default ``gpt-6-luna``, the only model the Decisions API supports).
- ``OPENAI_VISION_MODEL``: vision model that re-reads low-confidence OCR regions
  through the Responses API (default ``gpt-6-luna``).
- ``OPENAI_BASE_URL``: optional OpenAI-compatible endpoint for extraction and the
  VLM re-read (and the Decisions verifier when set; ``/v1/decisions`` is only
  served by OpenAI Cloud today). Unset keeps the SDK default.
- ``PAGE_LEGIBILITY_THRESHOLD``: normalized Decisions ``OCR_LEGIBILITY_SCORE``
  below which the critical fields of a scanned page are escalated even when
  their OCR confidence is high (default ``0.75``).
- ``PROMPT_INJECTION_RISK_THRESHOLD``: normalized Decisions
  ``PROMPT_INJECTION_RISK_SCORE`` at or above which a page is flagged as a
  suspected prompt injection (default ``0.5``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_OPENAI_MODEL: str = "gpt-6-luna"
"""Default OpenAI model used for structured extraction."""

DEFAULT_DECISIONS_MODEL: str = "gpt-6-luna"
"""Default model for the Decisions API (the only model it supports)."""

DEFAULT_VISION_MODEL: str = "gpt-6-luna"
"""Default vision model for the VLM re-read of low-confidence OCR regions."""


def _default_project_root() -> Path:
    """Return the repository root inferred from this module's location.

    Returns
    -------
    Path
        Absolute path two levels above ``src/core``.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline settings.

    Parameters
    ----------
    project_root : Path
        Repository root; every relative path is resolved against it.
    openai_api_key : str or None
        OpenAI credential. ``None`` selects the deterministic cache providers.
    openai_extraction_model : str
        Model used for live structured extraction (Responses API).
    openai_decisions_model : str
        Model used by the Tier 2 Decisions API verifier.
    openai_vision_model : str
        Vision model that re-reads low-confidence OCR regions (Responses API).
    openai_base_url : str or None
        Optional API base URL passed to ``openai.OpenAI(base_url=...)``;
        ``None`` keeps the SDK default endpoint.
    openai_extraction_temperature : float or None
        Sampling temperature for live extraction; ``None`` omits the parameter.
    audit_log_override : Path or None
        Explicit audit log location replacing ``logs/pipeline_audit.jsonl``.
    native_char_threshold : int
        Minimum clean alphanumeric characters for a page to be treated as native.
    ocr_confidence_threshold : float
        OCR region confidence below which the region is re-read by the vision
        model and the evidence it supports is escalated to Tier 2.
    verifier_confidence_threshold : float
        Calibrated predicate confidence below which human review is required.
    review_confidence_threshold : float
        Generic evidence confidence below which human review is required.
    fuzzy_match_threshold : float
        Minimum rapidfuzz similarity (0-100) for person entity matching.
    grounding_partial_ratio_threshold : float
        Minimum rapidfuzz partial ratio (0-100) for the grounding check.
    legibility_threshold : float
        Normalized page legibility score (``score / (levels - 1)``) below
        which the critical fields of a scanned page become escalation
        candidates regardless of their OCR confidence.
    injection_risk_threshold : float
        Normalized prompt-injection risk score at or above which a page is
        flagged and the document is routed to human review.

    Notes
    -----
    Derived paths are exposed as read-only properties so that a single
    ``project_root`` override relocates the whole layout (useful in tests).
    """

    project_root: Path = field(default_factory=_default_project_root)
    openai_api_key: Optional[str] = None
    openai_extraction_model: str = DEFAULT_OPENAI_MODEL
    openai_decisions_model: str = DEFAULT_DECISIONS_MODEL
    openai_vision_model: str = DEFAULT_VISION_MODEL
    openai_base_url: Optional[str] = None
    openai_extraction_temperature: Optional[float] = None
    audit_log_override: Optional[Path] = None
    native_char_threshold: int = 50
    ocr_confidence_threshold: float = 0.85
    verifier_confidence_threshold: float = 0.80
    review_confidence_threshold: float = 0.80
    fuzzy_match_threshold: float = 85.0
    grounding_partial_ratio_threshold: float = 90.0
    legibility_threshold: float = 0.75
    injection_risk_threshold: float = 0.5

    @property
    def openai_model(self) -> str:
        """str: Backward-compatible alias of ``openai_extraction_model``."""
        return self.openai_extraction_model

    @property
    def data_dir(self) -> Path:
        """Path: Root data directory (``data``)."""
        return self.project_root / "data"

    @property
    def samples_dir(self) -> Path:
        """Path: Synthetic sample PDFs directory (``data/samples``)."""
        return self.data_dir / "samples"

    @property
    def cache_dir(self) -> Path:
        """Path: SHA-256 keyed extraction cache directory (``data/cache``)."""
        return self.data_dir / "cache"

    @property
    def ground_truth_path(self) -> Path:
        """Path: Ground truth file (``data/ground_truth/ground_truth.json``)."""
        return self.data_dir / "ground_truth" / "ground_truth.json"

    @property
    def outputs_dir(self) -> Path:
        """Path: Final JSON outputs directory (``outputs``)."""
        return self.project_root / "outputs"

    @property
    def logs_dir(self) -> Path:
        """Path: Log directory (``logs``)."""
        return self.project_root / "logs"

    @property
    def audit_log_path(self) -> Path:
        """Path: JSON Lines audit trail (``logs/pipeline_audit.jsonl`` unless overridden)."""
        if self.audit_log_override is not None:
            return self.audit_log_override
        return self.logs_dir / "pipeline_audit.jsonl"

    @property
    def live_mode_available(self) -> bool:
        """bool: ``True`` when a non-empty OpenAI API key is configured."""
        return bool(self.openai_api_key)

    def cache_file_for(self, doc_sha256: str) -> Path:
        """Return the cache file path for a document hash.

        Parameters
        ----------
        doc_sha256 : str
            Hex-encoded SHA-256 digest of the document binary.

        Returns
        -------
        Path
            ``data/cache/{doc_sha256}.json``.
        """
        return self.cache_dir / f"{doc_sha256}.json"


def _parse_optional_float(name: str) -> Optional[float]:
    """Read an optional float environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    float or None
        Parsed value, or ``None`` when unset or empty.

    Raises
    ------
    ValueError
        If the value is set but is not a finite number.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


def _parse_unit_threshold(name: str, default: float) -> float:
    """Read an optional threshold in ``[0.0, 1.0]`` from the environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Value used when the variable is unset or empty.

    Returns
    -------
    float
        Parsed threshold.

    Raises
    ------
    ValueError
        If the value is not a finite number within ``[0.0, 1.0]``.
    """
    value = _parse_optional_float(name)
    if value is None:
        return default
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {value}")
    return value


def load_settings_from_env() -> Settings:
    """Build a fresh :class:`Settings` instance from environment variables.

    Returns
    -------
    Settings
        Settings reflecting the current process environment.

    Raises
    ------
    ValueError
        If ``OPENAI_EXTRACTION_TEMPERATURE`` is set to a non-numeric value,
        or a Decisions score threshold is outside ``[0.0, 1.0]``.

    Notes
    -----
    Empty-string environment values are treated as unset.
    """
    root_env = os.environ.get("ACTAS_PROJECT_ROOT", "").strip()
    audit_env = os.environ.get("ACTAS_AUDIT_LOG_PATH", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    extraction_model = (
        os.environ.get("OPENAI_EXTRACTION_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
    )
    decisions_model = os.environ.get("OPENAI_DECISIONS_MODEL", "").strip()
    vision_model = os.environ.get("OPENAI_VISION_MODEL", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    return Settings(
        project_root=Path(root_env).resolve() if root_env else _default_project_root(),
        openai_api_key=api_key or None,
        openai_extraction_model=extraction_model or DEFAULT_OPENAI_MODEL,
        openai_decisions_model=decisions_model or DEFAULT_DECISIONS_MODEL,
        openai_vision_model=vision_model or DEFAULT_VISION_MODEL,
        openai_base_url=base_url or None,
        openai_extraction_temperature=_parse_optional_float("OPENAI_EXTRACTION_TEMPERATURE"),
        audit_log_override=Path(audit_env) if audit_env else None,
        legibility_threshold=_parse_unit_threshold("PAGE_LEGIBILITY_THRESHOLD", 0.75),
        injection_risk_threshold=_parse_unit_threshold("PROMPT_INJECTION_RISK_THRESHOLD", 0.5),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings.

    Returns
    -------
    Settings
        Settings loaded once from the environment.

    Notes
    -----
    Call ``get_settings.cache_clear()`` after mutating the environment
    (for example in tests) to force a reload.
    """
    return load_settings_from_env()
