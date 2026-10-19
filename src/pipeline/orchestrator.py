"""End-to-end pipeline orchestration for corporate deed processing.

The orchestrator wires ingestion, page-level inspection, OCR, structured
extraction, temporal resolution, deterministic validation, two-tier
confidence escalation and persistence. It depends only on provider
abstractions; concrete implementations are chosen by
:func:`build_orchestrator`.

Notes
-----
Idempotency follows the SHA-256 cache contract: when ``data/cache/{sha}.json``
exists, OCR text, raw LLM extraction and verifier answers are served from it
by deterministic providers and no live call is made, even in live mode.
A cache miss in live mode runs the live providers and persists a new cache
payload so that the next run is idempotent.

A live run that meets a scanned page without the optional OCR engine fails
that document fast with an explicit ``[ERROR]`` naming ``uv sync --extra
ocr``; it never degrades silently. Pre-verified cache entries are never
overwritten, and live results are not persisted when grounding fails.
Value-in-quote failures do not block persistence because that check is
recomputed deterministically on every run.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rapidfuzz import fuzz

from src.core.config import Settings
from src.core.exceptions import (
    CacheMissError,
    IngestionError,
    OCRError,
    PDFInspectionError,
    PipelineError,
    VerifierError,
)
from src.core.logger import StatusTag, get_logger, log_event
from src.core.registry import ACTA_CONSTITUTIVA, DocumentRegistry, DocumentTypeSpec
from src.ingestion.hasher import compute_sha256
from src.ingestion.image_loader import is_image_path, load_document_bytes
from src.ingestion.pdf_inspector import PDFInspector, render_region_png
from src.pipeline.resolver import TemporalResolver
from src.pipeline.validators import (
    LIST_FIELDS,
    apply_confidence_routing,
    verify_capital_consistency,
    verify_grounding,
    verify_temporal_consistency,
    verify_value_in_quote,
)
from src.providers.interfaces import (
    BaseLLMProvider,
    BaseOCRProvider,
    BaseStorageRepository,
    BaseVerifierProvider,
    OCRRegion,
    OCRResult,
    PageContent,
    PageSource,
    VerifierResult,
)
from src.providers.llm_provider import (
    SYSTEM_PROMPT_SHA256,
    CachedLLMProvider,
    build_llm_provider,
)
from src.providers.ocr_provider import DeterministicOCRProvider, build_ocr_provider
from src.providers.verifier_provider import (
    DeterministicVerifierProvider,
    build_verifier_provider,
)
from src.schemas.deed_models import (
    AmendmentType,
    AuditMetadata,
    BoundingBox,
    ConfidenceSource,
    CorporateDeedOutput,
    Evidence,
    FieldStatus,
    FieldWithEvidence,
    RawDeedExtraction,
)
from src.storage.repository import FileSystemRepository

_LOGGER = get_logger("pipeline.orchestrator")

_JOB_ID_LENGTH: int = 12
_TEXT_MATCH_MIN_RATIO: float = 85.0
_PREDICATE_QUOTE_LIMIT: int = 200
_CAPITAL_FIELD: str = "capital_social_actual"
_PRE_VERIFIED_MODEL_VERSION: str = "pre-verified-cache-v1"
_MIN_REGION_OVERLAP_CHARS: int = 3
_REGION_SEPARATOR: str = " "


class DocumentProcessingError(PipelineError):
    """Raised when an unexpected non-pipeline error aborts one document.

    Notes
    -----
    Wraps arbitrary exceptions so callers only need to handle
    :class:`PipelineError`; the original exception is chained.
    """


@dataclass(frozen=True)
class ProviderBundle:
    """A coherent set of OCR, verifier and LLM providers.

    Parameters
    ----------
    ocr : BaseOCRProvider
        OCR provider for scanned pages.
    verifier : BaseVerifierProvider
        Escalation Tier 2 predicate verifier.
    llm : BaseLLMProvider
        Structured extraction provider.
    """

    ocr: BaseOCRProvider
    verifier: BaseVerifierProvider
    llm: BaseLLMProvider


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of processing one document.

    Parameters
    ----------
    acta_id : str
        Document identifier (file stem).
    doc_sha256 : str
        SHA-256 of the PDF binary.
    output_path : Path
        Written ``outputs/{acta_id}.json`` path.
    output : CorporateDeedOutput
        Final validated output.
    cache_hit : bool
        Whether the SHA-256 cache served the providers.
    duration_ms : float
        Total wall-clock time for the document.
    review_fields : list of str
        Contract fields that require human review.
    """

    acta_id: str
    doc_sha256: str
    output_path: Path
    output: CorporateDeedOutput
    cache_hit: bool
    duration_ms: float
    review_fields: list[str]


@dataclass(frozen=True)
class ProcessingFailure:
    """A document that could not be processed.

    Parameters
    ----------
    path : Path
        Source PDF path.
    error_type : str
        Exception class name.
    message : str
        Failure description.
    """

    path: Path
    error_type: str
    message: str


@dataclass
class _DocumentContext:
    """Mutable per-document working state.

    Parameters
    ----------
    job_id : str
        Job identifier.
    doc_sha256 : str
        Document hash.
    pages : dict of int to PageContent
        Inspected pages with OCR text filled in.
    ocr_results : dict of int to OCRResult
        OCR output per scanned page.
    verifier_log : list of dict
        Verifier answers collected for cache persistence.
    verified : dict
        Per-document memo of verifier results keyed by page, bbox and predicate.
    pdf_bytes : bytes
        PDF binary kept available for rendering verification crops.
    verifier_failures : int
        Number of predicates the verifier could not answer (refusal or error).
    """

    job_id: str
    doc_sha256: str
    pdf_bytes: bytes = b""
    verifier_failures: int = 0
    pages: dict[int, PageContent] = field(default_factory=dict)
    ocr_results: dict[int, OCRResult] = field(default_factory=dict)
    verifier_log: list[dict[str, Any]] = field(default_factory=list)
    verified: dict[tuple[int, tuple[float, ...], str], VerifierResult] = field(default_factory=dict)

    @property
    def page_texts(self) -> dict[int, str]:
        """dict of int to str: Page text used downstream (native or OCR)."""
        return {number: page.text for number, page in self.pages.items()}


@dataclass(frozen=True)
class _LocatedRegion:
    """OCR evidence location used for escalation decisions.

    Parameters
    ----------
    confidence : float
        OCR confidence covering the evidence.
    bbox : BoundingBox
        Region location in PDF points.
    """

    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class _EvidenceRef:
    """One evidence together with the fields that depend on it.

    Parameters
    ----------
    name : str
        Contract field name, or ``amendment_history[i]`` for an event.
    wrapper : FieldWithEvidence or None
        Owning field for field and list-item evidence; ``None`` for events.
    evidence : Evidence
        The evidence object (mutated in place by anchoring and escalation).
    dependents : tuple of FieldWithEvidence
        Fields flagged for review when this evidence stays unreliable.
    """

    name: str
    wrapper: Optional[FieldWithEvidence[Any]]
    evidence: Evidence
    dependents: tuple[FieldWithEvidence[Any], ...]


class LazyOCRProvider(BaseOCRProvider):
    """OCR provider that defers engine construction to the first page.

    Parameters
    ----------
    repository : BaseStorageRepository
        Repository forwarded to :func:`build_ocr_provider`.
    engine : str, default "rapidocr"
        Engine name forwarded to :func:`build_ocr_provider`.

    Notes
    -----
    Allows a live bundle to exist without failing when every document is
    served from the cache and the optional OCR engine is not installed.
    """

    def __init__(self, repository: BaseStorageRepository, engine: str = "rapidocr") -> None:
        """Store the construction arguments.

        Parameters
        ----------
        repository : BaseStorageRepository
            Cache repository.
        engine : str, default "rapidocr"
            Engine name.
        """
        self._repository = repository
        self._engine = engine
        self._provider: Optional[BaseOCRProvider] = None

    def extract_page(self, doc_sha256: str, page_number: int, image_png: bytes) -> OCRResult:
        """Recognise a page, building the engine on first use.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            1-indexed page number.
        image_png : bytes
            PNG rendering of the page.

        Returns
        -------
        OCRResult
            Recognised text and regions.

        Raises
        ------
        OCRError
            If the engine is unavailable or recognition fails.
        """
        if self._provider is None:
            self._provider = build_ocr_provider(self._repository, self._engine)
        return self._provider.extract_page(doc_sha256, page_number, image_png)


def _normalize_space(text: str) -> str:
    """Collapse whitespace and lowercase a text for matching.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Lowercase text with single spaces.
    """
    return " ".join(text.lower().split())


def _union_bbox(regions: list[OCRRegion]) -> BoundingBox:
    """Return the union bounding box of OCR regions.

    Parameters
    ----------
    regions : list of OCRRegion
        Non-empty list of regions.

    Returns
    -------
    BoundingBox
        Smallest box containing every region.
    """
    return BoundingBox(
        x1=min(region.bbox.x1 for region in regions),
        y1=min(region.bbox.y1 for region in regions),
        x2=max(region.bbox.x2 for region in regions),
        y2=max(region.bbox.y2 for region in regions),
    )


def _quote_span_regions(quote: str, regions: list[OCRRegion]) -> list[OCRRegion]:
    """Select the OCR regions that hold a normalized quote.

    Parameters
    ----------
    quote : str
        Normalized evidence quote.
    regions : list of OCRRegion
        Page regions in reading order.

    Returns
    -------
    list of OCRRegion
        Consecutive regions whose character ranges intersect the span that
        the quote aligns to in the joined page text; empty when the best
        alignment scores below the fuzzy threshold.

    Notes
    -----
    Region texts are normalized and joined with a single space.
    ``rapidfuzz.fuzz.partial_ratio_alignment`` locates the quote in the joined
    text, so unrelated short regions elsewhere on the page never contribute.
    A region must overlap the span by at least ``min(len(region), 3)``
    characters, which ignores one-character bleed at fuzzy alignment edges.
    """
    ranges: list[tuple[int, int, OCRRegion]] = []
    parts: list[str] = []
    cursor = 0
    for region in regions:
        text = _normalize_space(region.text)
        if not text:
            continue
        ranges.append((cursor, cursor + len(text), region))
        parts.append(text)
        cursor += len(text) + len(_REGION_SEPARATOR)
    if not ranges:
        return []
    alignment = fuzz.partial_ratio_alignment(quote, _REGION_SEPARATOR.join(parts))
    if alignment is None or float(alignment.score) < _TEXT_MATCH_MIN_RATIO:
        return []
    span_start, span_end = alignment.dest_start, alignment.dest_end
    selected: list[OCRRegion] = []
    for start, end, region in ranges:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap >= min(end - start, _MIN_REGION_OVERLAP_CHARS):
            selected.append(region)
    return selected


def locate_evidence_region(ocr: OCRResult, evidence: Evidence) -> Optional[_LocatedRegion]:
    """Find the OCR region(s) supporting an evidence quote.

    Parameters
    ----------
    ocr : OCRResult
        OCR output of the evidence page.
    evidence : Evidence
        Evidence to locate.

    Returns
    -------
    _LocatedRegion or None
        Minimum confidence over every region overlapping the evidence
        bounding box; when there is no box or no overlap, the minimum
        confidence over the consecutive regions the quote aligns to, with
        their union bounding box. ``None`` if nothing matches.

    Notes
    -----
    A multi-line quote spans several OCR lines; using the minimum keeps a
    single low-confidence line (for example an amount under a seal) from
    being masked by clean neighbouring lines. Regions are chosen by the
    character span of the quote inside the joined page text, found with
    ``rapidfuzz.fuzz.partial_ratio_alignment`` at 85 or above, so unrelated
    short regions elsewhere on the page do not affect the result.
    """
    if evidence.bounding_box is not None:
        overlapping: list[OCRRegion] = [
            region for region in ocr.regions if evidence.bounding_box.overlaps(region.bbox)
        ]
        if overlapping:
            return _LocatedRegion(
                confidence=min(region.confidence for region in overlapping),
                bbox=evidence.bounding_box,
            )
    quote = _normalize_space(evidence.source_quote)
    if not quote:
        return None
    matched = _quote_span_regions(quote, ocr.regions)
    if not matched:
        return None
    return _LocatedRegion(
        confidence=min(region.confidence for region in matched), bbox=_union_bbox(matched)
    )


class PipelineOrchestrator:
    """Coordinate the full document processing pipeline.

    Parameters
    ----------
    settings : Settings
        Pipeline settings and thresholds.
    repository : BaseStorageRepository
        Cache and output persistence.
    inspector : PDFInspector
        Page-level native/scanned router.
    resolver : TemporalResolver
        Temporal entity resolver.
    registry : DocumentRegistry
        Document type registry.
    cached_providers : ProviderBundle
        Deterministic providers serving cache hits.
    live_providers : ProviderBundle or None, default None
        Live providers used on cache misses (and ``force_refresh``). When
        ``None`` a cache miss raises :class:`CacheMissError`.

    Attributes
    ----------
    failures : list of ProcessingFailure
        Failures recorded by the most recent :meth:`process_all` call.

    Notes
    -----
    Depends only on abstractions; the concrete classes are selected by
    :func:`build_orchestrator`.
    """

    def __init__(
        self,
        settings: Settings,
        repository: BaseStorageRepository,
        inspector: PDFInspector,
        resolver: TemporalResolver,
        registry: DocumentRegistry,
        cached_providers: ProviderBundle,
        live_providers: Optional[ProviderBundle] = None,
    ) -> None:
        """Store the collaborators.

        Parameters
        ----------
        settings : Settings
            Pipeline settings.
        repository : BaseStorageRepository
            Persistence abstraction.
        inspector : PDFInspector
            Page inspector.
        resolver : TemporalResolver
            Temporal resolver.
        registry : DocumentRegistry
            Document type registry.
        cached_providers : ProviderBundle
            Deterministic providers for cache hits.
        live_providers : ProviderBundle or None, default None
            Live providers for cache misses.
        """
        self._settings = settings
        self._repository = repository
        self._inspector = inspector
        self._resolver = resolver
        self._registry = registry
        self._cached = cached_providers
        self._live = live_providers
        self.failures: list[ProcessingFailure] = []

    def process_all(
        self,
        paths: list[Path],
        *,
        document_type: str = ACTA_CONSTITUTIVA,
        force_refresh: bool = False,
    ) -> list[ProcessingResult]:
        """Process several documents, continuing past individual failures.

        Parameters
        ----------
        paths : list of Path
            PDF files to process, in order.
        document_type : str, default ACTA_CONSTITUTIVA
            Registered document type applied to every path.
        force_refresh : bool, default False
            Forwarded to :meth:`process_document`.

        Returns
        -------
        list of ProcessingResult
            Results of the documents that succeeded, in input order.

        Notes
        -----
        A failing document is logged as ``[ERROR]`` and recorded in
        :attr:`failures`; remaining documents are still processed. The
        caller decides whether failures change the exit status.
        """
        self.failures = []
        results: list[ProcessingResult] = []
        for path in paths:
            try:
                results.append(self.process_document(
                        path, document_type=document_type, force_refresh=force_refresh
                    ))
            except PipelineError as exc:
                self.failures.append(
                    ProcessingFailure(path=path, error_type=type(exc).__name__, message=exc.message)
                )
        return results

    def process_document(
        self,
        pdf_path: Path,
        *,
        document_type: str = ACTA_CONSTITUTIVA,
        force_refresh: bool = False,
    ) -> ProcessingResult:
        """Run the full pipeline for one PDF or one image.

        Parameters
        ----------
        pdf_path : Path
            Source PDF, or a PNG/JPG image treated as a one-page scanned
            document.
        document_type : str, default ACTA_CONSTITUTIVA
            Registered document type.
        force_refresh : bool, default False
            Ignore an existing cache entry and use the live providers when
            available (otherwise the cache still serves the providers and a
            WARN is logged).

        Returns
        -------
        ProcessingResult
            Output, location and review summary.

        Raises
        ------
        PipelineError
            On any processing failure (the class identifies the stage);
            unexpected exceptions are wrapped in
            :class:`DocumentProcessingError`.
        """
        return self.process_files(
            [pdf_path], document_type=document_type, force_refresh=force_refresh
        )

    def process_images(
        self,
        paths: list[Path],
        *,
        acta_id: str,
        document_type: str = ACTA_CONSTITUTIVA,
        force_refresh: bool = False,
    ) -> ProcessingResult:
        """Run the full pipeline for several images forming one document.

        Parameters
        ----------
        paths : list of Path
            PNG/JPG images in page order; image ``N`` becomes page ``N``.
        acta_id : str
            Document identifier used for the output and cache metadata.
        document_type : str, default ACTA_CONSTITUTIVA
            Registered document type.
        force_refresh : bool, default False
            Ignore an existing cache entry when live providers exist.

        Returns
        -------
        ProcessingResult
            Output, location and review summary.

        Raises
        ------
        PipelineError
            On any processing failure, including non-image paths.
        """
        return self.process_files(
            paths, acta_id=acta_id, document_type=document_type, force_refresh=force_refresh
        )

    def process_files(
        self,
        paths: list[Path],
        *,
        acta_id: Optional[str] = None,
        document_type: str = ACTA_CONSTITUTIVA,
        force_refresh: bool = False,
    ) -> ProcessingResult:
        """Run the full pipeline for one document made of one PDF or images.

        Parameters
        ----------
        paths : list of Path
            Exactly one PDF, or one or more PNG/JPG images in page order.
        acta_id : str or None, default None
            Document identifier; defaults to the stem of the first path.
        document_type : str, default ACTA_CONSTITUTIVA
            Registered document type.
        force_refresh : bool, default False
            Ignore an existing cache entry when live providers exist.

        Returns
        -------
        ProcessingResult
            Output, location and review summary.

        Raises
        ------
        PipelineError
            On any processing failure; unexpected exceptions are wrapped in
            :class:`DocumentProcessingError`.

        Notes
        -----
        The SHA-256 idempotency key is computed over the original input
        bytes (for images, never over the wrapped PDF), so the same photos
        always map to the same cache entry.
        """
        job_id = uuid.uuid4().hex[:_JOB_ID_LENGTH]
        context = _DocumentContext(job_id=job_id, doc_sha256="-")
        started = time.perf_counter()
        label = ", ".join(path.name for path in paths)
        try:
            return self._run(paths, acta_id, document_type, force_refresh, context, started)
        except PipelineError as exc:
            self._log_failure(exc, context, started)
            raise
        except Exception as exc:
            wrapped = DocumentProcessingError(
                f"unexpected {type(exc).__name__} while processing {label}: {exc}",
                {"path": label},
            )
            self._log_failure(exc, context, started)
            raise wrapped from exc

    def _log_failure(self, exc: Exception, context: _DocumentContext, started: float) -> None:
        """Emit the ``[ERROR]`` record for a failed document.

        Parameters
        ----------
        exc : Exception
            The raised exception.
        context : _DocumentContext
            Working state (job and hash).
        started : float
            ``perf_counter`` start of the document.
        """
        message = exc.message if isinstance(exc, PipelineError) else str(exc)
        details = dict(exc.details) if isinstance(exc, PipelineError) else {}
        details["error_type"] = type(exc).__name__
        log_event(
            _LOGGER,
            tag=StatusTag.ERROR,
            event="document_failed",
            stage="pipeline",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(started),
            details=details,
            message=f"{type(exc).__name__}: {message}",
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        """Return milliseconds elapsed since ``started``.

        Parameters
        ----------
        started : float
            ``time.perf_counter`` reading.

        Returns
        -------
        float
            Elapsed milliseconds rounded to three decimals.
        """
        return round((time.perf_counter() - started) * 1000.0, 3)

    def _run(
        self,
        paths: list[Path],
        acta_id_override: Optional[str],
        document_type: str,
        force_refresh: bool,
        context: _DocumentContext,
        started: float,
    ) -> ProcessingResult:
        """Execute every pipeline stage for one document.

        Parameters
        ----------
        paths : list of Path
            One PDF or one or more images.
        acta_id_override : str or None
            Explicit document identifier, or ``None`` for the first stem.
        document_type : str
            Registered document type.
        force_refresh : bool
            Whether to bypass the cache.
        context : _DocumentContext
            Working state, updated in place.
        started : float
            ``perf_counter`` start of the document.

        Returns
        -------
        ProcessingResult
            Final result.
        """
        spec = self._registry.get(document_type)
        stage_start = time.perf_counter()
        original_bytes, pdf_bytes, source_kind = self._load_source(paths)
        context.pdf_bytes = pdf_bytes
        context.doc_sha256 = compute_sha256(original_bytes)
        acta_id = acta_id_override or paths[0].stem
        source_name = ", ".join(path.name for path in paths)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="document_ingested",
            stage="ingestion",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={
                "acta_id": acta_id,
                "bytes": len(original_bytes),
                "document_type": document_type,
                "source_kind": source_kind,
                "image_count": len(paths) if source_kind == "image" else 0,
            },
            message=(
                f"{source_name} ingested ({len(original_bytes)} bytes, {source_kind}"
                + (f", {len(paths)} images" if source_kind == "image" else "")
                + ")"
            ),
        )

        cache_hit, providers = self._select_providers(acta_id, force_refresh, context)

        stage_start = time.perf_counter()
        inspected = self._inspector.inspect(
            pdf_bytes, job_id=context.job_id, doc_sha256=context.doc_sha256
        )
        self._fill_ocr(inspected, providers.ocr, context)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="inspection_completed",
            stage="inspection",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={
                "total_pages": len(inspected),
                "ocr_pages": len(context.ocr_results),
            },
            message=(
                f"{len(inspected)} pages inspected, "
                f"{len(inspected) - len(context.ocr_results)} native, "
                f"{len(context.ocr_results)} routed to OCR"
            ),
        )

        stage_start = time.perf_counter()
        raw, model_version = providers.llm.extract(
            context.doc_sha256, [context.pages[n] for n in sorted(context.pages)], document_type
        )
        raw_snapshot = raw.model_dump(by_alias=True, mode="json")
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="extraction_completed",
            stage="extraction",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={"model_version": model_version, "amendment_events": len(raw.amendment_history)},
            message=f"structured extraction completed ({model_version})",
        )

        stage_start = time.perf_counter()
        resolved = self._resolve(raw, spec, context)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="resolution_completed",
            stage="resolution",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={"temporal_resolution": spec.temporal_resolution},
            message="temporal resolution completed",
        )

        stage_start = time.perf_counter()
        self._anchor_evidence(resolved, context)
        failures = verify_grounding(
            resolved,
            context.page_texts,
            threshold=self._settings.grounding_partial_ratio_threshold,
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
        )
        value_failures = verify_value_in_quote(
            resolved, job_id=context.job_id, doc_sha256=context.doc_sha256
        )
        temporal_failures = verify_temporal_consistency(
            resolved, job_id=context.job_id, doc_sha256=context.doc_sha256
        )
        consistent = verify_capital_consistency(
            resolved, job_id=context.job_id, doc_sha256=context.doc_sha256
        )
        self._enforce_spec(resolved, spec, context)
        log_event(
            _LOGGER,
            tag=(
                StatusTag.OK
                if not failures and not value_failures and not temporal_failures and consistent
                else StatusTag.WARN
            ),
            event="validation_completed",
            stage="validation",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={
                "grounding_failures": len(failures),
                "value_in_quote_failures": len(value_failures),
                "temporal_failures": len(temporal_failures),
                "capital_consistent": consistent,
            },
            message=(
                f"validation completed: {len(failures)} grounding failures, "
                f"{len(value_failures)} value-in-quote failures, "
                f"{len(temporal_failures)} temporal failures, "
                f"capital {'consistent' if consistent else 'inconsistent'}"
            ),
        )

        stage_start = time.perf_counter()
        escalated = self._escalate(resolved, providers.verifier, context)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="escalation_completed",
            stage="escalation",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={"escalated_evidences": escalated},
            message=f"escalation completed ({escalated} evidences verified)",
        )

        self._route_amendment_confidence(resolved, context)
        apply_confidence_routing(
            resolved,
            self._settings.review_confidence_threshold,
            preserve_existing_reason=True,
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
        )

        if not cache_hit:
            if context.verifier_failures:
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="cache_not_written",
                    stage="persistence",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={"verifier_failures": context.verifier_failures},
                    message=f"live result for {acta_id} not cached: verifier could not answer",
                )
            elif failures:
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="cache_not_written",
                    stage="persistence",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={"grounding_failures": len(failures)},
                    message=f"live result for {acta_id} not cached: grounding validation failed",
                )
            else:
                self._persist_cache(
                    source_name, acta_id, document_type, model_version, raw_snapshot, context
                )

        return self._finalize(
            source_name, acta_id, spec, resolved, model_version, cache_hit, context, started
        )

    def _load_source(self, paths: list[Path]) -> tuple[bytes, bytes, str]:
        """Read the input files and produce hash and processing bytes.

        Parameters
        ----------
        paths : list of Path
            One PDF, or one or more PNG/JPG images.

        Returns
        -------
        tuple of (bytes, bytes, str)
            Original bytes for hashing, PDF bytes for processing and the
            source kind (``"pdf"`` or ``"image"``).

        Raises
        ------
        IngestionError
            If no path is given, images are mixed with PDFs, several PDFs
            are given, or a file cannot be read or decoded.
        """
        if not paths:
            raise IngestionError("no input files provided")
        image_flags = [is_image_path(path) for path in paths]
        if all(image_flags):
            original, wrapped = load_document_bytes(paths)
            return original, wrapped, "image"
        if any(image_flags) or len(paths) > 1:
            raise IngestionError(
                "a document is either one PDF or one or more images, not a mix",
                {"paths": [str(path) for path in paths]},
            )
        data = self._read_pdf(paths[0])
        return data, data, "pdf"

    @staticmethod
    def _read_pdf(pdf_path: Path) -> bytes:
        """Read the PDF binary.

        Parameters
        ----------
        pdf_path : Path
            Source PDF.

        Returns
        -------
        bytes
            File content.

        Raises
        ------
        IngestionError
            If the file is missing, unreadable or empty.
        """
        try:
            data = pdf_path.read_bytes()
        except OSError as exc:
            raise IngestionError(
                f"cannot read document: {pdf_path}", {"path": str(pdf_path), "error": str(exc)}
            ) from exc
        if not data:
            raise IngestionError(f"document is empty: {pdf_path}", {"path": str(pdf_path)})
        return data

    def _select_providers(
        self, acta_id: str, force_refresh: bool, context: _DocumentContext
    ) -> tuple[bool, ProviderBundle]:
        """Apply the idempotency rule and choose the provider bundle.

        Parameters
        ----------
        acta_id : str
            Document identifier (for messages).
        force_refresh : bool
            Whether to bypass the cache.
        context : _DocumentContext
            Working state (job and hash).

        Returns
        -------
        tuple of (bool, ProviderBundle)
            Cache hit flag and the providers to use.

        Raises
        ------
        CacheMissError
            On a cache miss when no live providers are configured.
        """
        stage_start = time.perf_counter()
        entry = self._repository.load_cache(context.doc_sha256)
        cached = entry is not None
        if cached and self._live is not None and not self._is_current_entry(entry):
            log_event(
                _LOGGER,
                tag=StatusTag.INFO,
                event="cache_stale",
                stage="idempotency",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"acta_id": acta_id},
                message=f"cache entry for {acta_id} was produced by another model or prompt; treated as miss",
            )
            cached = False
        if cached and not force_refresh:
            log_event(
                _LOGGER,
                tag=StatusTag.INFO,
                event="cache_hit",
                stage="idempotency",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                duration_ms=self._elapsed(stage_start),
                details={"acta_id": acta_id},
                message=f"cache hit for {acta_id}; OCR, extraction and verifier served from cache",
            )
            return True, self._cached
        if self._live is None:
            if cached:
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="force_refresh_unavailable",
                    stage="idempotency",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={"acta_id": acta_id},
                    message="force refresh requested without live providers; using cache",
                )
                return True, self._cached
            log_event(
                _LOGGER,
                tag=StatusTag.INFO,
                event="cache_miss",
                stage="idempotency",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                duration_ms=self._elapsed(stage_start),
                details={"acta_id": acta_id, "live": False},
                message=f"cache miss for {acta_id}",
            )
            raise CacheMissError(
                f"no cache entry for {acta_id} ({context.doc_sha256[:12]}); "
                "run 'actas-cli generate-samples' or set OPENAI_API_KEY and use --live",
                {"doc_sha256": context.doc_sha256, "acta_id": acta_id},
            )
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="cache_miss" if not cached else "cache_refresh",
            stage="idempotency",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={"acta_id": acta_id, "live": True, "force_refresh": force_refresh},
            message=f"{'cache miss' if not cached else 'forced refresh'} for {acta_id}; using live providers",
        )
        return False, self._live

    def _is_current_entry(self, entry: Optional[dict[str, Any]]) -> bool:
        """Decide whether a cache entry may serve the live path.

        Parameters
        ----------
        entry : dict or None
            Loaded cache entry.

        Returns
        -------
        bool
            ``True`` for pre-verified entries and for live entries whose
            prompt hash and model match the current configuration.
        """
        if entry is None:
            return False
        version = entry.get("model_version")
        if version == _PRE_VERIFIED_MODEL_VERSION:
            return True
        return (
            entry.get("prompt_sha256") == SYSTEM_PROMPT_SHA256
            and version == f"openai:{self._settings.openai_extraction_model}"
        )

    def _fill_ocr(
        self, inspected: list[PageContent], ocr: BaseOCRProvider, context: _DocumentContext
    ) -> None:
        """Run OCR on scanned pages and register all pages in the context.

        Parameters
        ----------
        inspected : list of PageContent
            Pages returned by the inspector.
        ocr : BaseOCRProvider
            OCR provider for scanned pages.
        context : _DocumentContext
            Working state, updated in place.

        Raises
        ------
        OCRError
            If a scanned page has no image or recognition fails.
        """
        for page in inspected:
            if page.source is not PageSource.SCANNED_PAGE:
                context.pages[page.page_number] = page
                continue
            if page.image_png is None:
                raise OCRError(
                    f"scanned page {page.page_number} has no rendered image",
                    {"page_number": page.page_number},
                )
            stage_start = time.perf_counter()
            result = ocr.extract_page(context.doc_sha256, page.page_number, page.image_png)
            context.ocr_results[page.page_number] = result
            context.pages[page.page_number] = page.model_copy(
                update={
                    "text": result.text,
                    "confidence_score": result.mean_confidence,
                    "confidence_source": ConfidenceSource.OCR_ENGINE,
                    "image_png": None,
                }
            )
            log_event(
                _LOGGER,
                tag=StatusTag.INFO,
                event="ocr_completed",
                stage="ocr",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                duration_ms=self._elapsed(stage_start),
                details={
                    "page_number": page.page_number,
                    "engine": result.engine,
                    "mean_confidence": result.mean_confidence,
                    "regions": len(result.regions),
                },
                message=(
                    f"page {page.page_number} OCR mean confidence {result.mean_confidence:.2f} "
                    f"({result.engine})"
                ),
            )

    def _resolve(
        self, raw: RawDeedExtraction, spec: DocumentTypeSpec, context: _DocumentContext
    ) -> RawDeedExtraction:
        """Run temporal resolution when the document type requires it.

        Parameters
        ----------
        raw : RawDeedExtraction
            Pre-resolution extraction.
        spec : DocumentTypeSpec
            Document type specification.
        context : _DocumentContext
            Working state (job and hash).

        Returns
        -------
        RawDeedExtraction
            Resolved copy, or a deep copy when resolution is not required.
        """
        if not spec.temporal_resolution:
            return raw.model_copy(deep=True)
        return self._resolver.resolve(raw, job_id=context.job_id, doc_sha256=context.doc_sha256)

    def _evidence_entries(self, model: RawDeedExtraction) -> list[_EvidenceRef]:
        """List every field, list-item and amendment event evidence.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction.

        Returns
        -------
        list of _EvidenceRef
            One entry for each field evidence, each list item evidence and
            each amendment event evidence. Event entries depend on the
            fields their act type determines.
        """
        entries: list[_EvidenceRef] = []
        for name in type(model).model_fields:
            if name == "amendment_history":
                continue
            wrapper: FieldWithEvidence[Any] = getattr(model, name)
            if wrapper.evidence is not None:
                entries.append(_EvidenceRef(name, wrapper, wrapper.evidence, (wrapper,)))
            if name in LIST_FIELDS and isinstance(wrapper.value, list):
                for item in wrapper.value:
                    item_evidence: Optional[Evidence] = getattr(item, "evidence", None)
                    if item_evidence is not None:
                        entries.append(_EvidenceRef(name, wrapper, item_evidence, (wrapper,)))
        for index, event in enumerate(model.amendment_history):
            entries.append(
                _EvidenceRef(
                    f"amendment_history[{index}]",
                    None,
                    event.evidence,
                    self._event_dependents(model, event.event_type),
                )
            )
        return entries

    @staticmethod
    def _event_dependents(
        model: RawDeedExtraction, event_type: AmendmentType
    ) -> tuple[FieldWithEvidence[Any], ...]:
        """Return the fields whose value derives from an act type.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction.
        event_type : AmendmentType
            Type of the amendment event.

        Returns
        -------
        tuple of FieldWithEvidence
            Revocations and grants affect representatives and powers, a
            capital increase affects the current capital, the incorporation
            affects the incorporation date.
        """
        if event_type in (AmendmentType.REVOCATION, AmendmentType.GRANT_OF_POWERS):
            return (model.representantes_legales, model.poderes)
        if event_type is AmendmentType.CAPITAL_INCREASE:
            return (model.capital_social_actual,)
        return (model.fecha_constitucion,)

    def _anchor_evidence(self, model: RawDeedExtraction, context: _DocumentContext) -> None:
        """Anchor evidence confidence to the actual extraction channel.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction (mutated in place).
        context : _DocumentContext
            Working state with pages and OCR results.

        Notes
        -----
        Evidence on native pages is set to ``1.0`` / ``NATIVE_PDF``. Evidence
        on scanned pages takes the confidence of its OCR region with source
        ``OCR_ENGINE`` and receives the region bounding box when missing, so
        live LLM output cannot self-assert confidence.
        """
        for ref in self._evidence_entries(model):
            evidence = ref.evidence
            page = context.pages.get(evidence.page_number)
            if page is None:
                continue
            if page.source is PageSource.NATIVE_PDF:
                evidence.confidence_score = 1.0
                evidence.confidence_source = ConfidenceSource.NATIVE_PDF
                continue
            ocr = context.ocr_results.get(evidence.page_number)
            located = locate_evidence_region(ocr, evidence) if ocr is not None else None
            if located is None:
                continue
            if evidence.bounding_box is None:
                evidence.bounding_box = located.bbox
            evidence.confidence_score = located.confidence
            evidence.confidence_source = ConfidenceSource.OCR_ENGINE

    def _enforce_spec(
        self, model: RawDeedExtraction, spec: DocumentTypeSpec, context: _DocumentContext
    ) -> None:
        """Enforce the document type specification on the extraction.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction (mutated in place).
        spec : DocumentTypeSpec
            Document type specification.
        context : _DocumentContext
            Working state (job and hash).

        Notes
        -----
        A field outside ``applicable_fields`` becomes ``no_aplicable`` with
        no value or evidence. A required field reported as ``extraido``
        without a value or without any evidence becomes ``no_encontrado``
        with its value and evidence cleared, and is flagged for human review.
        """
        for name in type(model).model_fields:
            if name == "amendment_history":
                continue
            wrapper: FieldWithEvidence[Any] = getattr(model, name)
            if not spec.is_applicable(name):
                wrapper.status = FieldStatus.NOT_APPLICABLE
                wrapper.value = None
                wrapper.evidence = None
                continue
            if not spec.is_required(name) or wrapper.status is not FieldStatus.EXTRACTED:
                continue
            items = wrapper.value if isinstance(wrapper.value, list) else []
            has_evidence = wrapper.evidence is not None or any(
                getattr(item, "evidence", None) is not None for item in items
            )
            if wrapper.value is not None and has_evidence:
                continue
            reason = (
                f"Required field {name} has no extracted value"
                if wrapper.value is None
                else f"Required field {name} has no supporting evidence"
            )
            wrapper.status = FieldStatus.NOT_FOUND
            wrapper.value = None
            wrapper.evidence = None
            wrapper.flag_for_review(reason)
            log_event(
                _LOGGER,
                tag=StatusTag.REVIEW_REQUIRED,
                event="required_field_missing",
                stage="validation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"field": name, "document_type": spec.document_type},
                message=f"{name}: {reason}",
            )

    def _build_predicate(
        self, name: str, wrapper: Optional[FieldWithEvidence[Any]], evidence: Evidence
    ) -> str:
        """Build the Tier 2 predicate for an evidence.

        Parameters
        ----------
        name : str
            Contract field name.
        wrapper : FieldWithEvidence or None
            The field containing the evidence; ``None`` for amendment events.
        evidence : Evidence
            Evidence to verify.

        Returns
        -------
        str
            Capital predicate for ``capital_social_actual``; otherwise a
            reading-accuracy predicate over the evidence quote.
        """
        if name == _CAPITAL_FIELD and wrapper is not None and wrapper.value is not None:
            amount = float(wrapper.value.monto_numerico)
            return (
                f"The capital social amount on page {evidence.page_number} "
                f"equals ${amount:,.2f} MXN"
            )
        quote = " ".join(evidence.source_quote.split())[:_PREDICATE_QUOTE_LIMIT]
        return f'The text on page {evidence.page_number} reads: "{quote}"'

    def _escalate(
        self, model: RawDeedExtraction, verifier: BaseVerifierProvider, context: _DocumentContext
    ) -> int:
        """Apply Escalation Tiers 1 and 2 to low-confidence OCR evidence.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction (mutated in place).
        verifier : BaseVerifierProvider
            Tier 2 predicate verifier.
        context : _DocumentContext
            Working state with pages, OCR results and verifier memo.

        Returns
        -------
        int
            Number of distinct predicates sent to the verifier.

        Notes
        -----
        Tier 1: the OCR region covering the evidence has confidence below
        ``ocr_confidence_threshold``. Tier 2: the predicate verifier answers
        with probability ``p``; the calibrated confidence ``|2p - 1|``
        replaces the evidence confidence, and a value below
        ``verifier_confidence_threshold`` flags the field for human review.
        """
        verified_count = 0
        for ref in self._evidence_entries(model):
            name, wrapper, evidence = ref.name, ref.wrapper, ref.evidence
            page = context.pages.get(evidence.page_number)
            ocr = context.ocr_results.get(evidence.page_number)
            if page is None or ocr is None or page.source is not PageSource.SCANNED_PAGE:
                continue
            located = locate_evidence_region(ocr, evidence)
            if located is None or located.confidence >= self._settings.ocr_confidence_threshold:
                continue
            bbox = evidence.bounding_box or located.bbox
            predicate = self._build_predicate(name, wrapper, evidence)
            key = (evidence.page_number, tuple(bbox.as_list()), predicate)
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="tier1_escalation",
                stage="escalation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={
                    "field": name,
                    "page_number": evidence.page_number,
                    "ocr_confidence": located.confidence,
                    "ocr_threshold": self._settings.ocr_confidence_threshold,
                    "bbox": bbox.as_list(),
                },
                message=(
                    f"{name}: OCR confidence {located.confidence:.2f} below "
                    f"{self._settings.ocr_confidence_threshold:.2f} on page {evidence.page_number}; "
                    "escalating to predicate verification"
                ),
            )
            result = context.verified.get(key)
            if result is None:
                try:
                    result = verifier.verify_predicate(
                        context.doc_sha256,
                        evidence.page_number,
                        predicate,
                        context.page_texts[evidence.page_number],
                        bbox,
                        image_png=self._render_crop(context, evidence.page_number, bbox),
                    )
                except VerifierError as exc:
                    self._route_verifier_failure(ref, located.confidence, bbox, exc, context)
                    continue
                context.verified[key] = result
                context.verifier_log.append(
                    {
                        "page_number": evidence.page_number,
                        "bbox": bbox.as_list(),
                        "predicate": predicate,
                        "probability": result.probability,
                        "engine": result.engine,
                    }
                )
                verified_count += 1
            evidence.bounding_box = bbox
            if evidence.grounding_verified:
                evidence.confidence_score = result.calibrated_confidence
            else:
                evidence.confidence_score = min(
                    evidence.confidence_score, result.calibrated_confidence
                )
            evidence.confidence_source = ConfidenceSource.DECISIONS_PREDICATE
            threshold = self._settings.verifier_confidence_threshold
            if result.calibrated_confidence < threshold:
                reason = (
                    f"Low OCR confidence ({located.confidence:.2f}) and Decisions predicate "
                    f"confidence ({result.calibrated_confidence:.2f} < {threshold:.2f}) due to "
                    f"overlapping notary seal on page {evidence.page_number} bbox "
                    f"[{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]."
                )
                for dependent in ref.dependents:
                    dependent.flag_for_review(reason)
                log_event(
                    _LOGGER,
                    tag=StatusTag.REVIEW_REQUIRED,
                    event="human_review_required",
                    stage="escalation",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={
                        "field": name,
                        "page_number": evidence.page_number,
                        "ocr_confidence": located.confidence,
                        "probability": result.probability,
                        "calibrated_confidence": result.calibrated_confidence,
                        "verifier_threshold": threshold,
                        "bbox": bbox.as_list(),
                    },
                    message=(
                        f"{name}: OCR {located.confidence:.2f}, p={result.probability:.2f}, "
                        f"calibrated {result.calibrated_confidence:.2f} < {threshold:.2f}"
                    ),
                )
        return verified_count

    def _render_crop(
        self, context: _DocumentContext, page_number: int, bbox: BoundingBox
    ) -> Optional[bytes]:
        """Render the padded region crop sent to the verifier.

        Parameters
        ----------
        context : _DocumentContext
            Working state holding the PDF bytes.
        page_number : int
            1-indexed page number.
        bbox : BoundingBox
            Region of interest in PDF points.

        Returns
        -------
        bytes or None
            PNG crop at 200 dpi with 12 pt padding, or ``None`` (with a WARN
            record) when the region cannot be rendered, in which case the
            verifier judges the OCR text alone.
        """
        if not context.pdf_bytes:
            return None
        try:
            return render_region_png(context.pdf_bytes, page_number, bbox)
        except PDFInspectionError as exc:
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="verification_crop_failed",
                stage="escalation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"page_number": page_number, "bbox": bbox.as_list(), "error": exc.message},
                message=f"cannot render crop on page {page_number}; verifying text only",
            )
            return None

    def _route_verifier_failure(
        self,
        ref: _EvidenceRef,
        ocr_confidence: float,
        bbox: BoundingBox,
        exc: VerifierError,
        context: _DocumentContext,
    ) -> None:
        """Flag the fields behind an evidence the verifier could not answer.

        Parameters
        ----------
        ref : _EvidenceRef
            Evidence and its dependent fields.
        ocr_confidence : float
            OCR confidence of the located region.
        bbox : BoundingBox
            Region of interest.
        exc : VerifierError
            Refusal or request failure reported by the verifier.
        context : _DocumentContext
            Working state (counts the failure).

        Notes
        -----
        The evidence keeps its OCR-derived confidence and the dependents are
        routed to human review with an explicit reason; the failure is never
        silent and the live result is not cached.
        """
        evidence = ref.evidence
        context.verifier_failures += 1
        evidence.bounding_box = bbox
        evidence.confidence_score = min(evidence.confidence_score, ocr_confidence)
        reason = (
            f"Low OCR confidence ({ocr_confidence:.2f}) and Decisions predicate verification "
            f"unavailable ({exc.message}) on page {evidence.page_number} bbox "
            f"[{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]."
        )
        for dependent in ref.dependents:
            dependent.flag_for_review(reason)
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="verifier_unavailable",
            stage="escalation",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            details={
                "field": ref.name,
                "page_number": evidence.page_number,
                "ocr_confidence": ocr_confidence,
                "bbox": bbox.as_list(),
                "error": exc.message,
            },
            message=f"{ref.name}: {reason}",
        )

    def _route_amendment_confidence(
        self, model: RawDeedExtraction, context: _DocumentContext
    ) -> list[str]:
        """Flag fields that depend on a low-confidence amendment event.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction (review flags mutated in place).
        context : _DocumentContext
            Working state (job and hash).

        Returns
        -------
        list of str
            Locations of the amendment events that were below the threshold.

        Notes
        -----
        Runs after escalation, so an event that the verifier did not lift
        above ``review_confidence_threshold`` flags its dependent fields:
        revocation and grant events flag ``representantes_legales`` and
        ``poderes``, capital increases flag ``capital_social_actual`` and the
        incorporation flags ``fecha_constitucion``.
        """
        threshold = self._settings.review_confidence_threshold
        flagged: list[str] = []
        for ref in self._evidence_entries(model):
            if ref.wrapper is not None or ref.evidence.confidence_score >= threshold:
                continue
            reason = (
                f"Low confidence score ({ref.evidence.confidence_score:.2f} < {threshold:.2f}) "
                f"on {ref.name} evidence, page {ref.evidence.page_number}"
            )
            for dependent in ref.dependents:
                dependent.flag_for_review(reason)
            flagged.append(ref.name)
            log_event(
                _LOGGER,
                tag=StatusTag.REVIEW_REQUIRED,
                event="amendment_low_confidence",
                stage="escalation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={
                    "event": ref.name,
                    "page_number": ref.evidence.page_number,
                    "confidence": ref.evidence.confidence_score,
                    "threshold": threshold,
                },
                message=f"{ref.name}: {reason}",
            )
        return flagged

    def _persist_cache(
        self,
        source_name: str,
        acta_id: str,
        document_type: str,
        model_version: str,
        raw_snapshot: dict[str, Any],
        context: _DocumentContext,
    ) -> None:
        """Persist a live run as a cache entry for future idempotent runs.

        Parameters
        ----------
        source_name : str
            Source file name(s) recorded in the cache entry.
        acta_id : str
            Document identifier.
        document_type : str
            Registered document type.
        model_version : str
            Model identifier reported by the LLM provider.
        raw_snapshot : dict
            JSON dump of the raw (pre-resolution) extraction.
        context : _DocumentContext
            Working state with OCR results and verifier answers.
        """
        stage_start = time.perf_counter()
        existing = self._repository.load_cache(context.doc_sha256)
        if existing is not None and existing.get("model_version") == _PRE_VERIFIED_MODEL_VERSION:
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="cache_overwrite_refused",
                stage="persistence",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"acta_id": acta_id},
                message=f"pre-verified cache entry for {acta_id} kept; live result not persisted",
            )
            return
        payload: dict[str, Any] = {
            "doc_sha256": context.doc_sha256,
            "document_filename": source_name,
            "acta_id": acta_id,
            "document_type": document_type,
            "model_version": model_version,
            "prompt_sha256": SYSTEM_PROMPT_SHA256,
            "raw_extraction": raw_snapshot,
            "ocr_pages": {
                str(number): result.model_dump(mode="json")
                for number, result in sorted(context.ocr_results.items())
            },
            "verifier_responses": context.verifier_log,
        }
        path = self._repository.save_cache(context.doc_sha256, payload)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="cache_written",
            stage="persistence",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={"path": str(path)},
            message=f"cache entry written for {acta_id}",
        )

    def _finalize(
        self,
        source_name: str,
        acta_id: str,
        spec: DocumentTypeSpec,
        resolved: RawDeedExtraction,
        model_version: str,
        cache_hit: bool,
        context: _DocumentContext,
        started: float,
    ) -> ProcessingResult:
        """Assemble, persist and report the final output.

        Parameters
        ----------
        source_name : str
            Source file name(s).
        acta_id : str
            Document identifier.
        spec : DocumentTypeSpec
            Document type specification.
        resolved : RawDeedExtraction
            Resolved and validated extraction.
        model_version : str
            Model identifier.
        cache_hit : bool
            Whether the cache served the providers.
        context : _DocumentContext
            Working state.
        started : float
            ``perf_counter`` start of the document.

        Returns
        -------
        ProcessingResult
            Final result.
        """
        review_fields = [
            name
            for name in spec.required_fields
            if getattr(resolved, name).requires_human_review
        ]
        native_count = sum(1 for page in context.pages.values() if page.source is PageSource.NATIVE_PDF)
        metadata = AuditMetadata(
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            document_filename=source_name,
            model_version=model_version,
            total_pages=len(context.pages),
            native_pages_count=native_count,
            ocr_pages_count=len(context.pages) - native_count,
            global_human_review_required=bool(review_fields),
            processing_time_ms=self._elapsed(started),
        )
        payload = resolved.model_dump(mode="python")
        output = CorporateDeedOutput.model_validate({**payload, "audit_metadata": metadata.model_dump()})
        stage_start = time.perf_counter()
        output_path = self._repository.save_output(acta_id, output)
        duration_ms = self._elapsed(started)
        revoked = sum(
            1 for event in output.amendment_history if event.event_type is AmendmentType.REVOCATION
        )
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED if review_fields else StatusTag.OK,
            event="document_completed",
            stage="persistence",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={
                "acta_id": acta_id,
                "output_path": str(output_path),
                "cache_hit": cache_hit,
                "review_fields": review_fields,
                "revocation_events": revoked,
                "total_duration_ms": duration_ms,
            },
            message=(
                f"{acta_id} processed in {duration_ms:.1f} ms; "
                + (
                    f"human review required for {', '.join(review_fields)}"
                    if review_fields
                    else "no human review required"
                )
            ),
        )
        return ProcessingResult(
            acta_id=acta_id,
            doc_sha256=context.doc_sha256,
            output_path=output_path,
            output=output,
            cache_hit=cache_hit,
            duration_ms=duration_ms,
            review_fields=review_fields,
        )


def build_orchestrator(settings: Settings, *, live: bool) -> PipelineOrchestrator:
    """Assemble a :class:`PipelineOrchestrator` with concrete providers.

    Parameters
    ----------
    settings : Settings
        Pipeline settings.
    live : bool
        Whether live execution is requested for cache misses.

    Returns
    -------
    PipelineOrchestrator
        Orchestrator whose cache hits are served by deterministic providers
        and, when ``live`` and an API key are available, whose cache misses
        use OpenAI providers with RapidOCR for scanned pages.

    Notes
    -----
    ``live`` without ``OPENAI_API_KEY`` logs a WARN and falls back to the
    deterministic cache behaviour (cache misses then raise
    :class:`CacheMissError`).
    """
    repository = FileSystemRepository(settings)
    cached = ProviderBundle(
        ocr=DeterministicOCRProvider(repository),
        verifier=DeterministicVerifierProvider(repository),
        llm=CachedLLMProvider(repository),
    )
    registry = DocumentRegistry.default()
    live_bundle: Optional[ProviderBundle] = None
    if live and settings.live_mode_available:
        live_bundle = ProviderBundle(
            ocr=LazyOCRProvider(repository, "rapidocr"),
            verifier=build_verifier_provider(settings, repository, True),
            llm=build_llm_provider(settings, repository, True, registry),
        )
    elif live:
        log_event(
            _LOGGER,
            tag=StatusTag.WARN,
            event="live_unavailable",
            stage="pipeline",
            message="live mode requested without OPENAI_API_KEY; using deterministic cache",
        )
    return PipelineOrchestrator(
        settings=settings,
        repository=repository,
        inspector=PDFInspector(settings),
        resolver=TemporalResolver(settings),
        registry=registry,
        cached_providers=cached,
        live_providers=live_bundle,
    )


__all__ = [
    "DocumentProcessingError",
    "LazyOCRProvider",
    "PipelineOrchestrator",
    "ProcessingFailure",
    "ProcessingResult",
    "ProviderBundle",
    "build_orchestrator",
    "locate_evidence_region",
]
