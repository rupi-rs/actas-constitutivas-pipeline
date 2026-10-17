"""Idempotency tests: hashing, deterministic samples and cache behaviour."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from src.core.config import Settings
from src.core.exceptions import CacheMissError, IngestionError
from src.generators.synthetic_deeds import generate_synthetic_deeds
from src.ingestion.hasher import compute_file_sha256, compute_sha256
from src.pipeline.orchestrator import PipelineOrchestrator, ProviderBundle, build_orchestrator
from src.pipeline.resolver import TemporalResolver
from src.core.registry import DocumentRegistry
from src.ingestion.pdf_inspector import PDFInspector
from src.providers.interfaces import (
    BaseLLMProvider,
    BaseOCRProvider,
    BaseVerifierProvider,
    OCRResult,
    PageContent,
    VerifierResult,
)
from src.providers.llm_provider import CachedLLMProvider
from src.providers.ocr_provider import DeterministicOCRProvider
from src.providers.verifier_provider import DeterministicVerifierProvider
from src.schemas.deed_models import BoundingBox, RawDeedExtraction
from src.storage.repository import FileSystemRepository
from tests.conftest import read_audit_records


class SpyLLM(BaseLLMProvider):
    """LLM provider delegating to a reference provider while counting calls."""

    def __init__(self, delegate: BaseLLMProvider) -> None:
        """Store the delegate.

        Parameters
        ----------
        delegate : BaseLLMProvider
            Provider that actually serves the extraction.
        """
        self._delegate = delegate
        self.calls = 0

    def extract(
        self, doc_sha256: str, pages: list[PageContent], document_type: str
    ) -> tuple[RawDeedExtraction, str]:
        """Count the call and delegate.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        pages : list of PageContent
            Page contents.
        document_type : str
            Registered document type.

        Returns
        -------
        tuple of (RawDeedExtraction, str)
            Raw extraction and model version.
        """
        self.calls += 1
        raw, _ = self._delegate.extract(doc_sha256, pages, document_type)
        return raw, "fake-live-model-v1"


class SpyOCR(BaseOCRProvider):
    """OCR provider delegating to a reference provider while counting calls."""

    def __init__(self, delegate: BaseOCRProvider) -> None:
        """Store the delegate.

        Parameters
        ----------
        delegate : BaseOCRProvider
            Provider that actually serves the OCR result.
        """
        self._delegate = delegate
        self.calls = 0

    def extract_page(self, doc_sha256: str, page_number: int, image_png: bytes) -> OCRResult:
        """Count the call and delegate.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            One-based page number.
        image_png : bytes
            Page image.

        Returns
        -------
        OCRResult
            OCR output.
        """
        self.calls += 1
        return self._delegate.extract_page(doc_sha256, page_number, image_png)


class SpyVerifier(BaseVerifierProvider):
    """Verifier delegating to a reference provider while counting calls."""

    def __init__(self, delegate: BaseVerifierProvider) -> None:
        """Store the delegate.

        Parameters
        ----------
        delegate : BaseVerifierProvider
            Provider that actually serves the verification.
        """
        self._delegate = delegate
        self.calls = 0

    def verify_predicate(
        self,
        doc_sha256: str,
        page_number: int,
        predicate: str,
        context_text: str,
        bbox: Optional[BoundingBox],
    ) -> VerifierResult:
        """Count the call and delegate.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            One-based page number.
        predicate : str
            Predicate text.
        context_text : str
            Page context.
        bbox : BoundingBox or None
            Evidence region.

        Returns
        -------
        VerifierResult
            Verification outcome.
        """
        self.calls += 1
        return self._delegate.verify_predicate(doc_sha256, page_number, predicate, context_text, bbox)


def build_spy_bundle(reference: FileSystemRepository) -> tuple[ProviderBundle, SpyOCR, SpyVerifier, SpyLLM]:
    """Build a provider bundle serving from a reference cache with spies.

    Parameters
    ----------
    reference : FileSystemRepository
        Repository holding the pre-verified cache entries.

    Returns
    -------
    tuple
        The bundle and its three spies (OCR, verifier, LLM).
    """
    ocr = SpyOCR(DeterministicOCRProvider(reference))
    verifier = SpyVerifier(DeterministicVerifierProvider(reference))
    llm = SpyLLM(CachedLLMProvider(reference))
    return ProviderBundle(ocr=ocr, verifier=verifier, llm=llm), ocr, verifier, llm


def build_custom_orchestrator(
    settings: Settings, cached: ProviderBundle, live: Optional[ProviderBundle]
) -> PipelineOrchestrator:
    """Assemble an orchestrator from explicit provider bundles.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    cached : ProviderBundle
        Providers used on cache hits.
    live : ProviderBundle or None
        Providers used on cache misses.

    Returns
    -------
    PipelineOrchestrator
        Ready-to-run orchestrator.
    """
    return PipelineOrchestrator(
        settings=settings,
        repository=FileSystemRepository(settings),
        inspector=PDFInspector(settings),
        resolver=TemporalResolver(settings),
        registry=DocumentRegistry.default(),
        cached_providers=cached,
        live_providers=live,
    )


def comparable_output(output_json: dict[str, Any]) -> dict[str, Any]:
    """Drop run-specific audit fields from a serialized output.

    Parameters
    ----------
    output_json : dict
        Serialized ``CorporateDeedOutput``.

    Returns
    -------
    dict
        Copy without ``job_id`` and ``processing_time_ms``.
    """
    copy: dict[str, Any] = json.loads(json.dumps(output_json))
    copy["audit_metadata"].pop("job_id")
    copy["audit_metadata"].pop("processing_time_ms")
    return copy


def test_compute_sha256_matches_hashlib() -> None:
    """The byte hasher returns the lowercase hashlib digest."""
    data = b"acta constitutiva"
    assert compute_sha256(data) == hashlib.sha256(data).hexdigest()
    assert len(compute_sha256(data)) == 64


def test_compute_sha256_is_stable_across_calls() -> None:
    """Hashing the same bytes repeatedly yields the same digest."""
    assert len({compute_sha256(b"stable") for _ in range(5)}) == 1


def test_compute_sha256_rejects_empty_payload() -> None:
    """An empty payload is rejected with IngestionError."""
    with pytest.raises(IngestionError):
        compute_sha256(b"")


def test_file_hash_equals_bytes_hash(sample_paths: dict[str, Path]) -> None:
    """Streaming file hashing equals hashing the file bytes."""
    for path in sample_paths.values():
        assert compute_file_sha256(path) == compute_sha256(path.read_bytes())


def test_file_hash_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises IngestionError."""
    with pytest.raises(IngestionError):
        compute_file_sha256(tmp_path / "missing.pdf")


def test_sample_regeneration_is_deterministic(
    tmp_path: Path, sample_paths: dict[str, Path]
) -> None:
    """Generating the samples again yields identical SHA-256 digests."""
    other = Settings(project_root=tmp_path / "regen")
    regenerated = generate_synthetic_deeds(other)
    for deed in regenerated:
        assert deed.sha256 == compute_file_sha256(sample_paths[deed.acta_id])
        assert deed.cache_path.name == f"{deed.sha256}.json"


def test_cache_hit_skips_live_providers(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """A second run is a cache hit and the live LLM is never invoked."""
    reference = FileSystemRepository(settings)
    cached, _, _, cached_llm = build_spy_bundle(reference)
    live, live_ocr, live_verifier, live_llm = build_spy_bundle(reference)
    orchestrator = build_custom_orchestrator(settings, cached, live)
    first = orchestrator.process_document(sample_paths["acta_a"])
    second = orchestrator.process_document(sample_paths["acta_a"])
    assert first.cache_hit is True
    assert second.cache_hit is True
    assert live_llm.calls == 0
    assert live_ocr.calls == 0
    assert live_verifier.calls == 0
    assert cached_llm.calls == 2


@pytest.mark.parametrize("acta_id", ["acta_a", "acta_b", "acta_c"])
def test_repeated_runs_produce_identical_outputs(
    settings: Settings, sample_paths: dict[str, Path], acta_id: str
) -> None:
    """Two runs yield identical outputs ignoring job_id and processing time."""
    orchestrator = build_orchestrator(settings, live=False)
    first = orchestrator.process_document(sample_paths[acta_id])
    first_json = json.loads(first.output_path.read_text(encoding="utf-8"))
    second = orchestrator.process_document(sample_paths[acta_id])
    second_json = json.loads(second.output_path.read_text(encoding="utf-8"))
    assert first.cache_hit and second.cache_hit
    assert first.doc_sha256 == second.doc_sha256
    assert first.output.audit_metadata.job_id != second.output.audit_metadata.job_id
    assert comparable_output(first_json) == comparable_output(second_json)


def test_cache_miss_without_live_raises(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """Without a cache entry and without live providers CacheMissError is raised."""
    sha = compute_file_sha256(sample_paths["acta_a"])
    settings.cache_file_for(sha).unlink()
    orchestrator = build_orchestrator(settings, live=False)
    with pytest.raises(CacheMissError):
        orchestrator.process_document(sample_paths["acta_a"])
    assert not settings.outputs_dir.joinpath("acta_a.json").exists()


def test_process_all_records_cache_miss_failure(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """process_all keeps going after a cache miss and records the failure."""
    settings.cache_file_for(compute_file_sha256(sample_paths["acta_a"])).unlink()
    orchestrator = build_orchestrator(settings, live=False)
    results = orchestrator.process_all([sample_paths["acta_a"], sample_paths["acta_c"]])
    assert [result.acta_id for result in results] == ["acta_c"]
    assert [failure.error_type for failure in orchestrator.failures] == ["CacheMissError"]


@pytest.mark.parametrize("acta_id", ["acta_a", "acta_b"])
def test_live_miss_persists_cache_and_second_run_hits(
    settings: Settings, tmp_path: Path, sample_paths: dict[str, Path], acta_id: str
) -> None:
    """A live run on a cache miss writes a cache entry that the next run uses."""
    reference = FileSystemRepository(settings)
    sha = compute_file_sha256(sample_paths[acta_id])
    reference_payload = json.loads(settings.cache_file_for(sha).read_text(encoding="utf-8"))
    # Move the pre-verified entry out of the live cache so the first run misses
    settings.cache_file_for(sha).unlink()
    live_repo = _StaticRepository(settings, reference_payload)
    live, live_ocr, live_verifier, live_llm = build_spy_bundle(live_repo)
    cached, _, _, cached_llm = build_spy_bundle(FileSystemRepository(settings))
    orchestrator = build_custom_orchestrator(settings, cached, live)

    first = orchestrator.process_document(sample_paths[acta_id])
    assert first.cache_hit is False
    assert live_llm.calls == 1
    assert settings.cache_file_for(sha).is_file()
    persisted = json.loads(settings.cache_file_for(sha).read_text(encoding="utf-8"))
    assert persisted["doc_sha256"] == sha
    assert persisted["acta_id"] == acta_id
    assert persisted["model_version"] == "fake-live-model-v1"
    assert persisted["raw_extraction"] == reference_payload["raw_extraction"]
    if acta_id == "acta_b":
        assert live_ocr.calls == 1
        assert live_verifier.calls >= 1
        assert set(persisted["ocr_pages"]) == {"2"}
        assert persisted["verifier_responses"]

    second = orchestrator.process_document(sample_paths[acta_id])
    assert second.cache_hit is True
    assert live_llm.calls == 1
    assert cached_llm.calls == 1
    first_json = json.loads(first.output_path.read_text(encoding="utf-8"))
    second_json = json.loads(second.output_path.read_text(encoding="utf-8"))
    first_json["audit_metadata"]["model_version"] = second_json["audit_metadata"]["model_version"]
    assert comparable_output(first_json) == comparable_output(second_json)
    events = [record["event"] for record in read_audit_records(settings.audit_log_path)]
    assert "cache_miss" in events and "cache_written" in events and "cache_hit" in events


class _StaticRepository(FileSystemRepository):
    """Repository serving one fixed cache payload regardless of the hash."""

    def __init__(self, settings: Settings, payload: dict[str, Any]) -> None:
        """Store the fixed payload.

        Parameters
        ----------
        settings : Settings
            Isolated settings.
        payload : dict
            Cache payload returned by every ``load_cache`` call.
        """
        super().__init__(settings)
        self._payload = payload

    def load_cache(self, doc_sha256: str) -> Optional[dict[str, Any]]:
        """Return the fixed payload.

        Parameters
        ----------
        doc_sha256 : str
            Ignored document hash.

        Returns
        -------
        dict
            The fixed cache payload.
        """
        return self._payload
