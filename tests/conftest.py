"""Shared pytest fixtures for the Actas pipeline test suite.

Notes
-----
Every fixture is offline and isolated: samples are generated once per
session inside a pytest temporary directory and copied per test, so the real
``data/``, ``outputs/`` and ``logs/`` directories are never touched.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.core.config import Settings, get_settings
from src.core.logger import configure_logging, reset_logging
from src.generators.synthetic_deeds import GeneratedDeed, generate_synthetic_deeds
from src.schemas.deed_models import RawDeedExtraction

ACTA_IDS: tuple[str, ...] = ("acta_a", "acta_b", "acta_c")
"""Identifiers of the three synthetic deeds."""


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Clear the cached settings accessor around every test.

    Yields
    ------
    None
        Control to the test.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def generated_workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[GeneratedDeed]]:
    """Generate the synthetic deeds once per session.

    Notes
    -----
    Autouse so that generation (which resets logging) always runs before any
    per-test logging fixture is installed.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest factory for session-scoped temporary directories.

    Returns
    -------
    tuple of (Path, list of GeneratedDeed)
        Workspace root and the generated deed descriptors.
    """
    root = tmp_path_factory.mktemp("workspace")
    configure_logging(audit_log_path=root / "generation_audit.jsonl", console=False)
    try:
        deeds = generate_synthetic_deeds(Settings(project_root=root))
    finally:
        reset_logging()
    return root, deeds


@pytest.fixture
def audit_log_path(tmp_path: Path) -> Iterator[Path]:
    """Route logging to a temporary audit file for the duration of a test.

    Parameters
    ----------
    tmp_path : Path
        Per-test temporary directory.

    Yields
    ------
    Path
        JSON Lines audit log location (created lazily on first record).
    """
    path = tmp_path / "logs" / "audit.jsonl"
    configure_logging(audit_log_path=path)
    yield path
    reset_logging()


@pytest.fixture(autouse=True)
def isolated_logging(audit_log_path: Path) -> Path:
    """Guarantee that no test writes to the real audit log.

    Parameters
    ----------
    audit_log_path : Path
        Temporary audit log fixture.

    Returns
    -------
    Path
        The temporary audit log path.
    """
    return audit_log_path


@pytest.fixture
def settings(tmp_path: Path, generated_workspace: tuple[Path, list[GeneratedDeed]]) -> Settings:
    """Build per-test settings over a private copy of the generated workspace.

    Parameters
    ----------
    tmp_path : Path
        Per-test temporary directory.
    generated_workspace : tuple of (Path, list of GeneratedDeed)
        Session workspace with generated samples.

    Returns
    -------
    Settings
        Settings rooted in an isolated copy of ``data/``.
    """
    root = tmp_path / "project"
    shutil.copytree(generated_workspace[0] / "data", root / "data")
    return Settings(project_root=root, audit_log_override=tmp_path / "logs" / "audit.jsonl")


@pytest.fixture
def sample_paths(settings: Settings) -> dict[str, Path]:
    """Map each acta identifier to its generated PDF in the test workspace.

    Parameters
    ----------
    settings : Settings
        Isolated settings.

    Returns
    -------
    dict of str to Path
        PDF path per acta identifier.
    """
    return {acta_id: settings.samples_dir / f"{acta_id}.pdf" for acta_id in ACTA_IDS}


@pytest.fixture
def cache_payloads(settings: Settings, sample_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Load the pre-verified cache payload of every sample.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    sample_paths : dict of str to Path
        PDF path per acta identifier.

    Returns
    -------
    dict of str to dict
        Parsed cache JSON per acta identifier.
    """
    import hashlib

    payloads: dict[str, dict[str, Any]] = {}
    for acta_id, path in sample_paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payloads[acta_id] = json.loads(settings.cache_file_for(digest).read_text(encoding="utf-8"))
    return payloads


@pytest.fixture
def raw_extractions(cache_payloads: dict[str, dict[str, Any]]) -> dict[str, RawDeedExtraction]:
    """Parse the cached raw (pre-resolution) extraction of every sample.

    Parameters
    ----------
    cache_payloads : dict of str to dict
        Parsed cache JSON per acta identifier.

    Returns
    -------
    dict of str to RawDeedExtraction
        Fresh model instances per acta identifier.
    """
    return {
        acta_id: RawDeedExtraction.model_validate(payload["raw_extraction"])
        for acta_id, payload in cache_payloads.items()
    }


def read_audit_records(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON Lines audit log.

    Parameters
    ----------
    path : Path
        Audit log location.

    Returns
    -------
    list of dict
        One parsed record per line; empty when the file does not exist.
    """
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
