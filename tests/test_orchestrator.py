"""Orchestrator tests: evidence location, escalation caps, spec enforcement, cache policy."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pytest
from typer.testing import CliRunner

from src.cli.main import _say, app
from src.core.config import Settings
from src.core.logger import StatusTag
from src.core.registry import DocumentRegistry, DocumentTypeSpec
from src.ingestion.hasher import compute_file_sha256
from src.pipeline.orchestrator import (
    PipelineOrchestrator,
    ProviderBundle,
    _DocumentContext,
    locate_evidence_region,
)
from src.providers.interfaces import (
    BaseVerifierProvider,
    OCRRegion,
    OCRResult,
    PageContent,
    PageSource,
    VerifierResult,
)
from src.providers.llm_provider import SYSTEM_PROMPT_SHA256, CachedLLMProvider
from src.schemas.deed_models import (
    BoundingBox,
    ConfidenceSource,
    Evidence,
    FieldStatus,
    RawDeedExtraction,
)
from src.storage.repository import FileSystemRepository
from tests.conftest import read_audit_records
from tests.test_idempotency import (
    SpyLLM,
    _StaticRepository,
    build_custom_orchestrator,
    build_spy_bundle,
)


class FixedVerifier(BaseVerifierProvider):
    """Verifier returning a fixed probability and counting calls."""

    def __init__(self, probability: float) -> None:
        """Store the probability.

        Parameters
        ----------
        probability : float
            Probability of YES returned for every predicate.
        """
        self._probability = probability
        self.calls = 0

    def verify_predicate(
        self,
        doc_sha256: str,
        page_number: int,
        predicate: str,
        context_text: str,
        bbox: Optional[BoundingBox],
    ) -> VerifierResult:
        """Return the fixed result.

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
            Result with ``|2p - 1|`` confidence.
        """
        self.calls += 1
        return VerifierResult(
            predicate=predicate,
            probability=self._probability,
            calibrated_confidence=abs(2 * self._probability - 1),
            engine="fixed",
        )


def _two_line_ocr() -> OCRResult:
    """Build an OCR page whose second line has low confidence.

    Returns
    -------
    OCRResult
        Two regions, confidences 0.99 and 0.55.
    """
    regions = [
        OCRRegion(text="Capital social de", bbox=BoundingBox(x1=50, y1=100, x2=300, y2=120), confidence=0.99),
        OCRRegion(text="$7,000,000.00 MXN", bbox=BoundingBox(x1=50, y1=125, x2=300, y2=145), confidence=0.55),
    ]
    return OCRResult(
        page_number=1,
        text="Capital social de\n$7,000,000.00 MXN",
        mean_confidence=0.77,
        regions=regions,
        engine="test",
    )


def _quote_evidence(grounded: bool) -> Evidence:
    """Build evidence quoting both OCR lines.

    Parameters
    ----------
    grounded : bool
        Value of ``grounding_verified``.

    Returns
    -------
    Evidence
        Evidence with a two-line quote.
    """
    return Evidence(
        page_number=1,
        source_quote="Capital social de $7,000,000.00 MXN",
        confidence_score=1.0 if grounded else 0.4,
        confidence_source=ConfidenceSource.OCR_ENGINE,
        grounding_verified=grounded,
    )


def _scanned_context(ocr: OCRResult) -> _DocumentContext:
    """Build a context with one scanned page.

    Parameters
    ----------
    ocr : OCRResult
        OCR output stored for page 1.

    Returns
    -------
    _DocumentContext
        Working state.
    """
    page = PageContent(
        page_number=1,
        source=PageSource.SCANNED_PAGE,
        text=ocr.text,
        clean_char_count=0,
        confidence_score=ocr.mean_confidence,
        confidence_source=ConfidenceSource.OCR_ENGINE,
    )
    context = _DocumentContext(job_id="job", doc_sha256="a" * 64)
    context.pages[1] = page
    context.ocr_results[1] = ocr
    return context


def test_locate_uses_minimum_confidence_and_union_bbox() -> None:
    """A low-confidence second line is not masked by the clean first line."""
    located = locate_evidence_region(_two_line_ocr(), _quote_evidence(True))
    assert located is not None
    assert located.confidence == 0.55
    assert (located.bbox.x1, located.bbox.y1, located.bbox.x2, located.bbox.y2) == (50, 100, 300, 145)


def test_locate_bbox_overlap_uses_minimum() -> None:
    """The bounding-box path also takes the minimum confidence."""
    evidence = _quote_evidence(True)
    evidence.bounding_box = BoundingBox(x1=40, y1=90, x2=310, y2=150)
    located = locate_evidence_region(_two_line_ocr(), evidence)
    assert located is not None and located.confidence == 0.55


def test_two_line_quote_triggers_escalation_and_keeps_grounding_cap(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """Escalation happens for a low second line and never lifts the grounding cap.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    ocr = _two_line_ocr()
    for grounded, expected in ((True, 0.9), (False, 0.4)):
        raw = raw_extractions["acta_b"].model_copy(deep=True)
        raw.capital_social_actual.evidence = _quote_evidence(grounded)
        verifier = FixedVerifier(0.95)
        cached, _, _, _ = build_spy_bundle(FileSystemRepository(settings))
        bundle = ProviderBundle(ocr=cached.ocr, verifier=verifier, llm=cached.llm)
        orchestrator = build_custom_orchestrator(settings, bundle, None)
        context = _scanned_context(ocr)
        orchestrator._escalate(raw, verifier, context)
        assert verifier.calls >= 1
        assert raw.capital_social_actual.evidence is not None
        assert raw.capital_social_actual.evidence.confidence_score == pytest.approx(expected)
        assert raw.capital_social_actual.evidence.confidence_source is ConfidenceSource.DECISIONS_PREDICATE


def test_enforce_spec_marks_missing_required_and_inapplicable_fields(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """Required fields without value or evidence become no_encontrado; others no_aplicable.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    raw = raw_extractions["acta_a"].model_copy(deep=True)
    raw.razon_social.value = None
    raw.fecha_constitucion.evidence = None
    spec = DocumentTypeSpec(
        document_type="mini",
        display_name="Mini",
        required_fields=("razon_social", "fecha_constitucion"),
        applicable_fields=("razon_social", "fecha_constitucion", "notario"),
    )
    cached, _, _, _ = build_spy_bundle(FileSystemRepository(settings))
    orchestrator = build_custom_orchestrator(settings, cached, None)
    orchestrator._enforce_spec(raw, spec, _DocumentContext(job_id="job", doc_sha256="a" * 64))
    assert raw.razon_social.status is FieldStatus.NOT_FOUND
    assert raw.razon_social.requires_human_review is True
    assert "no extracted value" in (raw.razon_social.review_reason or "")
    assert raw.fecha_constitucion.status is FieldStatus.NOT_FOUND
    assert "no supporting evidence" in (raw.fecha_constitucion.review_reason or "")
    assert raw.socios.status is FieldStatus.NOT_APPLICABLE and raw.socios.value is None
    assert raw.notario.status is FieldStatus.EXTRACTED


def test_unknown_document_type_cli_exits_with_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--document-type with an unregistered value exits 1 with an [ERROR] line.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ACTAS_PROJECT_ROOT", str(settings.project_root))
    result = CliRunner().invoke(app, ["process-all", "--document-type", "nonexistent"])
    assert result.exit_code == 1
    assert "[ERROR]" in result.output


def test_cli_help_lists_only_pipeline_commands() -> None:
    """The CLI help lists only the documented pipeline commands."""
    result = CliRunner().invoke(app, ["--help"])
    generate = CliRunner().invoke(app, ["generate-samples", "--help"])
    assert "study" not in (result.output + generate.output).lower()


def test_say_redacts_pii(capsys: pytest.CaptureFixture[str]) -> None:
    """Status lines mask RFC values before printing.

    Parameters
    ----------
    capsys : object
        Pytest stdout and stderr capture fixture.
    """
    _say(StatusTag.ERROR, "failure for PERJ800101ABC")
    out = capsys.readouterr().out
    assert "PERJ800101ABC" not in out and "PERJ******ABC" in out


def _live_setup(
    settings: Settings, sample: Path, llm_mutator: Any = None, reference: Optional[FileSystemRepository] = None
) -> tuple[PipelineOrchestrator, SpyLLM]:
    """Build an orchestrator whose live bundle replays the sample cache.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    sample : Path
        PDF whose cache entry feeds the live providers.
    llm_mutator : callable or None, optional
        Hook applied to the live raw extraction.
    reference : FileSystemRepository or None, optional
        Repository feeding the live providers; defaults to the settings repository.

    Returns
    -------
    tuple
        Orchestrator and the live LLM spy.
    """
    reference = reference if reference is not None else FileSystemRepository(settings)
    live, _, _, live_llm = build_spy_bundle(reference)
    if llm_mutator is not None:
        live = ProviderBundle(ocr=live.ocr, verifier=live.verifier, llm=_MutatingLLM(live_llm, llm_mutator))
    cached, _, _, _ = build_spy_bundle(reference)
    return build_custom_orchestrator(settings, cached, live), live_llm


class _MutatingLLM(CachedLLMProvider):
    """LLM wrapper applying a mutation to the extraction."""

    def __init__(self, delegate: SpyLLM, mutator: Any) -> None:
        """Store the delegate and mutator.

        Parameters
        ----------
        delegate : SpyLLM
            Spy serving the extraction.
        mutator : callable
            Function mutating the raw extraction in place.
        """
        self._delegate = delegate
        self._mutator = mutator

    def extract(
        self, doc_sha256: str, pages: list[PageContent], document_type: str
    ) -> tuple[RawDeedExtraction, str]:
        """Delegate and mutate.

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
            Mutated extraction and model version.
        """
        raw, version = self._delegate.extract(doc_sha256, pages, document_type)
        self._mutator(raw)
        return raw, version


def test_force_refresh_never_overwrites_pre_verified_entry(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A live force-refresh keeps the tracked pre-verified cache entry intact.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    sample = sample_paths["acta_a"]
    cache_file = settings.cache_file_for(compute_file_sha256(sample))
    before = cache_file.read_text(encoding="utf-8")
    orchestrator, live_llm = _live_setup(settings, sample)
    result = orchestrator.process_document(sample, force_refresh=True)
    assert result.cache_hit is False and live_llm.calls == 1
    assert cache_file.read_text(encoding="utf-8") == before
    events = [r["event"] for r in read_audit_records(settings.audit_log_path)]
    assert "cache_overwrite_refused" in events


def test_live_result_with_grounding_failure_is_not_cached(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A live result whose quote is absent from the page is not persisted.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    sample = sample_paths["acta_a"]
    cache_file = settings.cache_file_for(compute_file_sha256(sample))
    reference = _StaticRepository(settings, json.loads(cache_file.read_text(encoding="utf-8")))
    cache_file.unlink()

    def corrupt(raw: RawDeedExtraction) -> None:
        """Replace the razon_social quote with text absent from every page.

        Parameters
        ----------
        raw : RawDeedExtraction
            Extraction mutated in place.
        """
        assert raw.razon_social.evidence is not None
        raw.razon_social.evidence.source_quote = "texto que no existe en ninguna pagina del documento"

    orchestrator, _ = _live_setup(settings, sample, corrupt, reference)
    orchestrator.process_document(sample)
    assert not cache_file.exists()
    events = [r["event"] for r in read_audit_records(settings.audit_log_path)]
    assert "cache_not_written" in events


def test_stale_live_entry_is_a_miss_and_current_entry_is_a_hit(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """On the live path a different prompt hash or model is a miss.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    settings = replace(settings, openai_model="fake-live-model")
    sample = sample_paths["acta_a"]
    cache_file = settings.cache_file_for(compute_file_sha256(sample))
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    model_version = f"openai:{settings.openai_model}"
    payload["model_version"] = model_version
    payload["prompt_sha256"] = "0" * 64
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    orchestrator, live_llm = _live_setup(settings, sample)
    assert orchestrator.process_document(sample).cache_hit is False
    assert live_llm.calls == 1
    persisted = json.loads(cache_file.read_text(encoding="utf-8"))
    assert persisted["prompt_sha256"] == SYSTEM_PROMPT_SHA256
    assert persisted["model_version"] == "openai:fake-live-model"
    persisted["model_version"] = model_version
    cache_file.write_text(json.dumps(persisted), encoding="utf-8")
    assert orchestrator.process_document(sample).cache_hit is True


def test_registry_default_contains_acta_constitutiva() -> None:
    """The CLI default document type is registered."""
    assert "acta_constitutiva" in DocumentRegistry.default()
