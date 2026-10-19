"""Orchestrator tests: evidence location, escalation caps, spec enforcement, cache policy."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pytest
from PIL import Image
from typer.testing import CliRunner

from src.cli.main import _revocation_lines, _say, app
from src.core.config import Settings
from src.core.exceptions import VerifierError
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
    AmendmentEvent,
    AmendmentType,
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
        self.images: list[Optional[bytes]] = []

    def verify_predicate(
        self,
        doc_sha256: str,
        page_number: int,
        predicate: str,
        context_text: str,
        bbox: Optional[BoundingBox],
        *,
        image_png: Optional[bytes] = None,
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
        image_png : bytes or None, default None
            Region crop forwarded by the orchestrator.

        Returns
        -------
        VerifierResult
            Result with ``|2p - 1|`` confidence.
        """
        self.calls += 1
        self.images.append(image_png)
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


class RefusingVerifier(BaseVerifierProvider):
    """Verifier that always fails as a Decisions refusal would."""

    def verify_predicate(
        self,
        doc_sha256: str,
        page_number: int,
        predicate: str,
        context_text: str,
        bbox: Optional[BoundingBox],
        *,
        image_png: Optional[bytes] = None,
    ) -> VerifierResult:
        """Raise the refusal error.

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
        image_png : bytes or None, default None
            Region crop.

        Returns
        -------
        VerifierResult
            Never returns.

        Raises
        ------
        VerifierError
            Always.
        """
        raise VerifierError("Decisions API refused to answer the verification predicate")


def test_verifier_refusal_routes_field_to_review(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """A verifier refusal flags the field with an explicit reason instead of aborting.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    raw = raw_extractions["acta_b"].model_copy(deep=True)
    raw.capital_social_actual.evidence = _quote_evidence(True)
    raw.capital_social_actual.requires_human_review = False
    raw.capital_social_actual.review_reason = None
    cached, _, _, _ = build_spy_bundle(FileSystemRepository(settings))
    verifier = RefusingVerifier()
    bundle = ProviderBundle(ocr=cached.ocr, verifier=verifier, llm=cached.llm)
    orchestrator = build_custom_orchestrator(settings, bundle, None)
    context = _scanned_context(_two_line_ocr())
    assert orchestrator._escalate(raw, verifier, context) == 0
    assert context.verifier_failures == 1
    assert raw.capital_social_actual.requires_human_review is True
    reason = raw.capital_social_actual.review_reason or ""
    assert "Decisions predicate verification unavailable" in reason
    assert "refused" in reason
    assert raw.capital_social_actual.evidence is not None
    assert raw.capital_social_actual.evidence.confidence_score == pytest.approx(0.55)
    events = [r["event"] for r in read_audit_records(settings.audit_log_path)]
    assert "verifier_unavailable" in events


def test_scanned_capital_escalation_sends_region_crop(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """Acta B escalation passes a 200 dpi PNG crop of the padded capital bbox.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    cached, _, spy_verifier, _ = build_spy_bundle(FileSystemRepository(settings))
    orchestrator = build_custom_orchestrator(settings, cached, None)
    result = orchestrator.process_document(sample_paths["acta_b"])
    assert result.output.capital_social_actual.requires_human_review is True
    assert spy_verifier.calls == 1
    crop = spy_verifier.images[0]
    assert crop is not None and crop.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(crop)) as image:
        width, height = image.size
    scale = 200 / 72
    assert width == pytest.approx((520 - 72 + 24) * scale, abs=2)
    assert height == pytest.approx((455 - 410 + 24) * scale, abs=2)


def test_locate_ignores_stray_short_regions() -> None:
    """Unrelated short regions elsewhere on the page do not affect the result."""
    regions = [
        OCRRegion(text="M.N.", bbox=BoundingBox(x1=400, y1=10, x2=440, y2=20), confidence=0.40),
        OCRRegion(
            text="Capital social de $7,000,000.00 MXN",
            bbox=BoundingBox(x1=50, y1=100, x2=300, y2=120),
            confidence=0.95,
        ),
        OCRRegion(text="DE C.V.", bbox=BoundingBox(x1=400, y1=500, x2=450, y2=510), confidence=0.30),
    ]
    ocr = OCRResult(
        page_number=1,
        text="\n".join(region.text for region in regions),
        mean_confidence=0.5,
        regions=regions,
        engine="test",
    )
    located = locate_evidence_region(ocr, _quote_evidence(True))
    assert located is not None
    assert located.confidence == 0.95
    assert (located.bbox.x1, located.bbox.y1, located.bbox.x2, located.bbox.y2) == (50, 100, 300, 120)


def _revocation_page_ocr(quote: str, confidence: float) -> OCRResult:
    """Build a one-line OCR page holding a revocation quote.

    Parameters
    ----------
    quote : str
        Text of the single OCR region.
    confidence : float
        Region confidence.

    Returns
    -------
    OCRResult
        OCR page 1.
    """
    region = OCRRegion(
        text=quote, bbox=BoundingBox(x1=50, y1=100, x2=500, y2=120), confidence=confidence
    )
    return OCRResult(
        page_number=1, text=quote, mean_confidence=confidence, regions=[region], engine="test"
    )


def test_low_confidence_revocation_event_flags_dependent_fields(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """A revocation read at 0.60 on a scanned page flags representatives and powers.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    raw = raw_extractions["acta_c"].model_copy(deep=True)
    event = next(e for e in raw.amendment_history if e.event_type is AmendmentType.REVOCATION)
    quote = event.evidence.source_quote
    event.evidence.page_number = 1
    event.evidence.bounding_box = None
    for other in raw.amendment_history:
        if other is not event:
            other.evidence.page_number = 2
    context = _scanned_context(_revocation_page_ocr(quote, 0.60))
    verifier = FixedVerifier(0.55)
    cached, _, _, _ = build_spy_bundle(FileSystemRepository(settings))
    bundle = ProviderBundle(ocr=cached.ocr, verifier=verifier, llm=cached.llm)
    orchestrator = build_custom_orchestrator(settings, bundle, None)
    for field_name in ("representantes_legales", "poderes", "capital_social_actual"):
        wrapper = getattr(raw, field_name)
        wrapper.requires_human_review = False
        wrapper.review_reason = None
    orchestrator._anchor_evidence(raw, context)
    assert event.evidence.confidence_score == pytest.approx(0.60)
    assert event.evidence.confidence_source is ConfidenceSource.OCR_ENGINE
    assert event.evidence.bounding_box is not None
    orchestrator._escalate(raw, verifier, context)
    orchestrator._route_amendment_confidence(raw, context)
    assert verifier.calls >= 1
    assert event.evidence.confidence_source is ConfidenceSource.DECISIONS_PREDICATE
    assert raw.representantes_legales.requires_human_review is True
    assert raw.poderes.requires_human_review is True
    assert raw.capital_social_actual.requires_human_review is False


def _revocation_event(
    subject: str, matched: Optional[str], raw: RawDeedExtraction
) -> AmendmentEvent:
    """Build a resolved revocation event from a sample event.

    Parameters
    ----------
    subject : str
        ``sujeto_afectado`` value.
    matched : str or None
        ``matched_canonical_entity`` value.
    raw : RawDeedExtraction
        Extraction supplying a revocation event template.

    Returns
    -------
    AmendmentEvent
        Event copy with the given subject and match.
    """
    template = next(e for e in raw.amendment_history if e.event_type is AmendmentType.REVOCATION)
    event = template.model_copy(deep=True)
    event.sujeto_afectado = subject
    event.matched_canonical_entity = matched
    event.fuzzy_match_score = 100.0 if matched else None
    return event


def test_cli_unapplied_revocation_prints_actual_reason(
    raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """An unapplied revocation prints the representatives review reason.

    Parameters
    ----------
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    event = _revocation_event("Pedro Inexistente", None, raw_extractions["acta_c"])
    lines = _revocation_lines("acta_c", event, "Ambiguous revocation match: 'Pedro' vs 'Pedro X'")
    assert len(lines) == 1
    tag, message = lines[0]
    assert tag is StatusTag.WARN
    assert "Ambiguous revocation match" in message and "unmatched revocation" not in message


def test_cli_multi_subject_revocation_prints_one_line_per_person(
    raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """Each matched person gets a REVOKED line and unapplied subjects a WARN line.

    Parameters
    ----------
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    event = _revocation_event(
        "Juan Perez Lopez, Maria Garcia Soto y Luis Hernandez Ruiz",
        "Juan Perez Lopez; Maria Garcia Soto",
        raw_extractions["acta_c"],
    )
    lines = _revocation_lines("acta_c", event, "Revocation of 'Luis Hernandez Ruiz' did not match")
    revoked = [m for t, m in lines if t is StatusTag.REVOKED]
    warned = [m for t, m in lines if t is StatusTag.WARN]
    assert len(revoked) == 2
    assert "Juan Perez Lopez" in revoked[0] and "Maria Garcia Soto" in revoked[1]
    assert len(warned) == 1 and "Luis Hernandez Ruiz" in warned[0]
    assert "did not match" in warned[0]


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
    assert raw.razon_social.value is None and raw.razon_social.evidence is None
    assert "no extracted value" in (raw.razon_social.review_reason or "")
    assert raw.fecha_constitucion.status is FieldStatus.NOT_FOUND
    assert "no supporting evidence" in (raw.fecha_constitucion.review_reason or "")
    assert raw.fecha_constitucion.value is None and raw.fecha_constitucion.evidence is None
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


def test_live_result_with_value_in_quote_failure_is_still_cached(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A value-in-quote failure is recomputed every run and does not block the cache.

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

    def mismatch(raw: RawDeedExtraction) -> None:
        """Make the capital value disagree with its own quote.

        Parameters
        ----------
        raw : RawDeedExtraction
            Extraction mutated in place.
        """
        assert raw.capital_social_actual.value is not None
        raw.capital_social_actual.value.monto_numerico = 123456789.0

    orchestrator, _ = _live_setup(settings, sample, mismatch, reference)
    orchestrator.process_document(sample)
    assert cache_file.exists()
    events = [r["event"] for r in read_audit_records(settings.audit_log_path)]
    assert "cache_not_written" not in events
    assert "cache_written" in events


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
    settings = replace(settings, openai_extraction_model="fake-live-model")
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
