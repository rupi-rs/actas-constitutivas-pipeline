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
- ``OPENAI_MODEL``: OpenAI model identifier (default ``gpt-4o-2024-08-06``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_OPENAI_MODEL: str = "gpt-4o-2024-08-06"
"""Default OpenAI model used for structured extraction and verification."""


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
    openai_model : str
        OpenAI model identifier.
    audit_log_override : Path or None
        Explicit audit log location replacing ``logs/pipeline_audit.jsonl``.
    native_char_threshold : int
        Minimum clean alphanumeric characters for a page to be treated as native.
    ocr_confidence_threshold : float
        OCR region confidence below which Escalation Tier 2 is triggered.
    verifier_confidence_threshold : float
        Calibrated predicate confidence below which human review is required.
    review_confidence_threshold : float
        Generic evidence confidence below which human review is required.
    fuzzy_match_threshold : float
        Minimum rapidfuzz similarity (0-100) for person entity matching.
    grounding_partial_ratio_threshold : float
        Minimum rapidfuzz partial ratio (0-100) for the grounding check.

    Notes
    -----
    Derived paths are exposed as read-only properties so that a single
    ``project_root`` override relocates the whole layout (useful in tests).
    """

    project_root: Path = field(default_factory=_default_project_root)
    openai_api_key: Optional[str] = None
    openai_model: str = DEFAULT_OPENAI_MODEL
    audit_log_override: Optional[Path] = None
    native_char_threshold: int = 50
    ocr_confidence_threshold: float = 0.85
    verifier_confidence_threshold: float = 0.80
    review_confidence_threshold: float = 0.80
    fuzzy_match_threshold: float = 85.0
    grounding_partial_ratio_threshold: float = 90.0

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


def load_settings_from_env() -> Settings:
    """Build a fresh :class:`Settings` instance from environment variables.

    Returns
    -------
    Settings
        Settings reflecting the current process environment.

    Notes
    -----
    Empty-string environment values are treated as unset.
    """
    root_env = os.environ.get("ACTAS_PROJECT_ROOT", "").strip()
    audit_env = os.environ.get("ACTAS_AUDIT_LOG_PATH", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    return Settings(
        project_root=Path(root_env).resolve() if root_env else _default_project_root(),
        openai_api_key=api_key or None,
        openai_model=model or DEFAULT_OPENAI_MODEL,
        audit_log_override=Path(audit_env) if audit_env else None,
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
