"""End-to-end pipeline orchestration for corporate deed processing.

The orchestrator wires ingestion, page-level inspection, OCR, the VLM
re-read of low-confidence OCR regions, structured extraction, temporal
resolution, deterministic validation, two-tier confidence escalation and
persistence. It depends only on provider
abstractions; concrete implementations are chosen by
:func:`build_orchestrator`.

Notes
-----
Scanned pages follow a read-correct-extract-verify order: Tier 1 local OCR
reads the page; every region below ``ocr_confidence_threshold`` is cropped
and re-read by a vision model; a legible, non-empty re-read that differs
from the OCR text replaces it in the page text; structured extraction then
runs on the corrected text; finally the Tier 2 Decisions predicate verifies
the extracted value on the same crop.

The OpenAI Decisions API is used in one place only: the ``predicate`` that
verifies what the vision model re-read. On a scanned page (including image
inputs) every critical field of the document type always goes through the
VLM re-read of its own evidence region and then the Decisions predicate on
the same crop, independently of OCR confidence: local OCR engines can
silently drop characters hidden under a seal while reporting high
confidence for what remains. Non-critical fields on scanned pages are
re-read and verified only when their OCR region confidence is below the
threshold. Native pages never reach the vision model or Decisions.

Human review is decided per field from facts about its own value: its
calibrated ``|2p - 1|`` below ``verifier_confidence_threshold``, a re-read
of its own region judged ``legible=false``, a deterministic validation
failure, or a vision or verifier failure. A shared full-page re-read never
flags a field by itself.

Every page passes the offline regex prompt-injection pre-filter before
extraction (no API call); a match adds a document-level security flag,
forces global human review and prevents auto-approval of the fields on
that page. A required critical field that ends up ``no_encontrado``, or a
value-in-quote or capital consistency failure on a scanned page, triggers
one bounded VLM re-read of the relevant region (or full page) and at most
one re-extraction with the patched text.

Idempotency follows the SHA-256 cache contract: when ``data/cache/{sha}.json``
exists, OCR text, VLM re-reads, raw LLM extraction and verifier answers are served from it
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
    StorageError,
    VerifierError,
    VisionReadError,
)
from src.core.logger import StatusTag, get_logger, log_event
from src.core.registry import ACTA_CONSTITUTIVA, DocumentRegistry, DocumentTypeSpec
from src.ingestion.hasher import compute_sha256
from src.ingestion.image_loader import is_image_path, load_document_bytes
from src.ingestion.pdf_inspector import PDFInspector, page_bbox, render_region_png
from src.pipeline.injection_guard import detect_injection_patterns
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
    BaseVisionReader,
    OCRRegion,
    OCRResult,
    PageContent,
    PageSource,
    VerifierResult,
    VisionReadResult,
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
from src.providers.vision_provider import DeterministicVLMProvider, build_vision_provider
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
_READING_SUMMARY_LIMIT: int = 120
_REGION_COVER_TOLERANCE: float = 1.0
_MEMBER_MIN_CHARS: int = 6
_MEMBER_MIN_RATIO: float = 90.0
_MEMBER_FULL_COVERAGE: float = 0.8
_PAGE_LEVEL_TEXT_COVERAGE: float = 0.9

TRIGGER_OCR_CONFIDENCE: str = "ocr_confidence"
"""Re-read trigger: OCR region confidence below ``ocr_confidence_threshold``."""

TRIGGER_CRITICAL_FIELD: str = "critical_field"
"""Re-read trigger: critical field on a scanned page (always re-read and verified)."""

TRIGGER_MISSING_FIELD: str = "missing_field"
"""Re-read trigger: required critical field ``no_encontrado`` on a scanned document."""

TRIGGER_VALUE_INCONSISTENCY: str = "value_inconsistency"
"""Re-read trigger: value-in-quote or capital consistency failure on a scanned page."""

SECURITY_FLAG_INJECTION: str = "prompt_injection_suspected"
"""Security flag set when the regex pre-filter matches a page."""


class DocumentProcessingError(PipelineError):
    """Raised when an unexpected non-pipeline error aborts one document.

    Notes
    -----
    Wraps arbitrary exceptions so callers only need to handle
    :class:`PipelineError`; the original exception is chained.
    """


@dataclass(frozen=True)
class ProviderBundle:
    """A coherent set of OCR, vision, verifier and LLM providers.

    Parameters
    ----------
    ocr : BaseOCRProvider
        OCR provider for scanned pages.
    verifier : BaseVerifierProvider
        Escalation Tier 2 predicate verifier.
    llm : BaseLLMProvider
        Structured extraction provider.
    vision : BaseVisionReader or None, default None
        Vision reader for low-confidence OCR regions and critical-field
        regions of scanned pages. ``None`` disables the re-read (logged as
        WARN); evidence then goes straight to the Decisions predicate.
    """

    ocr: BaseOCRProvider
    verifier: BaseVerifierProvider
    llm: BaseLLMProvider
    vision: Optional[BaseVisionReader] = None


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
    security_flags : list of str, default []
        Document-level security flags (for example
        ``"prompt_injection_suspected"``).
    """

    acta_id: str
    doc_sha256: str
    output_path: Path
    output: CorporateDeedOutput
    cache_hit: bool
    duration_ms: float
    review_fields: list[str]
    security_flags: list[str] = field(default_factory=list)


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
    vision_failures : int
        Number of low-confidence regions the vision reader could not re-read.
    raw_ocr_results : dict of int to OCRResult
        Tier 1 OCR output before any VLM correction (persisted in the cache).
    rereads : list of _RegionReread
        VLM re-read outcome of every re-read region (low OCR confidence,
        critical field, missing field or value inconsistency).
    vlm_log : list of dict
        VLM re-reads collected for cache persistence.
    acta_id : str
        Document identifier used to name persisted crops.
    crop_names : set of str
        Crop file names already written for this document.
    suspicious_pages : dict of int to str
        Pages suspected of prompt injection, with the reason.
    security_flags : list of str
        Document-level security flags, in the order they were raised.
    reextracted : bool
        ``True`` once the single bounded re-extraction has run.
    page_boxes : dict of int to BoundingBox
        Memo of full-page bounding boxes.

    Notes
    -----
    ``ocr_results`` holds the corrected OCR (region texts patched by legible
    VLM readings, confidences unchanged) used to locate evidence.
    """

    job_id: str
    doc_sha256: str
    pdf_bytes: bytes = b""
    verifier_failures: int = 0
    vision_failures: int = 0
    pages: dict[int, PageContent] = field(default_factory=dict)
    ocr_results: dict[int, OCRResult] = field(default_factory=dict)
    raw_ocr_results: dict[int, OCRResult] = field(default_factory=dict)
    rereads: list[_RegionReread] = field(default_factory=list)
    vlm_log: list[dict[str, Any]] = field(default_factory=list)
    acta_id: str = "document"
    crop_names: set[str] = field(default_factory=set)
    verifier_log: list[dict[str, Any]] = field(default_factory=list)
    verified: dict[tuple[int, tuple[float, ...], str], VerifierResult] = field(default_factory=dict)
    suspicious_pages: dict[int, str] = field(default_factory=dict)
    security_flags: list[str] = field(default_factory=list)
    reextracted: bool = False
    page_boxes: dict[int, BoundingBox] = field(default_factory=dict)

    @property
    def page_texts(self) -> dict[int, str]:
        """dict of int to str: Page text used downstream (native or OCR)."""
        return {number: page.text for number, page in self.pages.items()}


@dataclass(frozen=True)
class _RegionReread:
    """VLM re-read outcome of one low-confidence OCR region.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    bbox : BoundingBox
        Region location in PDF points.
    ocr_text : str
        Tier 1 OCR reading of the region.
    ocr_confidence : float
        Tier 1 OCR confidence of the region.
    result : VisionReadResult or None
        Vision reading, or ``None`` when the re-read failed.
    error : str or None
        Failure description when the re-read failed.
    applied : bool
        ``True`` when the legible VLM reading replaced a differing OCR text.
    trigger : str, default TRIGGER_OCR_CONFIDENCE
        Why the region was re-read (OCR confidence, critical field,
        missing field or value inconsistency).
    """

    page_number: int
    bbox: BoundingBox
    ocr_text: str
    ocr_confidence: float
    result: Optional[VisionReadResult]
    error: Optional[str]
    applied: bool
    trigger: str = TRIGGER_OCR_CONFIDENCE


@dataclass
class _ExtractionPass:
    """Result of one extraction, resolution and validation pass.

    Parameters
    ----------
    raw : RawDeedExtraction
        Pre-resolution extraction.
    model_version : str
        Model identifier reported by the LLM provider.
    raw_snapshot : dict
        JSON dump of ``raw`` taken before resolution.
    resolved : RawDeedExtraction
        Resolved, anchored and validated extraction.
    grounding_failures : list of str
        Grounding failures.
    value_failures : list of str
        Fields whose value is not in their quote.
    temporal_failures : list of str
        Temporal inconsistencies.
    capital_consistent : bool
        Result of the capital number-versus-words check.
    """

    raw: RawDeedExtraction
    model_version: str
    raw_snapshot: dict[str, Any]
    resolved: RawDeedExtraction
    grounding_failures: list[str]
    value_failures: list[str]
    temporal_failures: list[str]
    capital_consistent: bool


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


def _format_bbox(bbox: BoundingBox) -> str:
    """Format a bounding box for review reasons.

    Parameters
    ----------
    bbox : BoundingBox
        Box to format.

    Returns
    -------
    str
        ``[x1, y1, x2, y2]`` with one decimal per coordinate.
    """
    return f"[{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]"


_SPANISH_MONTHS: tuple[str, ...] = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)


def _date_in_words(iso_date: str) -> str:
    """Render an ISO date the way Mexican deeds write it.

    Parameters
    ----------
    iso_date : str
        Date in ``YYYY-MM-DD`` form.

    Returns
    -------
    str
        ``"15 de marzo de 2021"``, or the input unchanged when it is not a valid ISO date.
    """
    parts = iso_date.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or not 1 <= int(parts[1]) <= 12:
        return iso_date
    return f"{int(parts[2])} de {_SPANISH_MONTHS[int(parts[1]) - 1]} de {parts[0]}"


def _summarize_reading(text: Optional[str]) -> str:
    """Quote a reading for a review reason, truncated and whitespace-collapsed.

    Parameters
    ----------
    text : str or None
        Reading to summarize.

    Returns
    -------
    str
        Double-quoted reading (``"null"`` when absent), at most 120 characters.
    """
    if text is None:
        return "null"
    collapsed = " ".join(text.split())
    if len(collapsed) > _READING_SUMMARY_LIMIT:
        collapsed = collapsed[: _READING_SUMMARY_LIMIT - 3] + "..."
    return f'"{collapsed}"'


def _patch_ocr_result(ocr: OCRResult, replacements: dict[int, str]) -> OCRResult:
    """Replace region texts of an OCR result and rebuild its page text.

    Parameters
    ----------
    ocr : OCRResult
        Tier 1 OCR output.
    replacements : dict of int to str
        New text per region index.

    Returns
    -------
    OCRResult
        Copy with patched region texts (confidences unchanged) and page text.

    Notes
    -----
    When the page text is the newline join of the regions (both shipped
    engines), it is rebuilt from the patched regions; otherwise each
    original region text is replaced once in place.
    """
    regions = [
        region.model_copy(update={"text": replacements[index]}) if index in replacements else region
        for index, region in enumerate(ocr.regions)
    ]
    if ocr.text == "\n".join(region.text for region in ocr.regions):
        text = "\n".join(region.text for region in regions)
    else:
        text = ocr.text
        for index, new_text in replacements.items():
            text = text.replace(ocr.regions[index].text, new_text, 1)
    return ocr.model_copy(update={"regions": regions, "text": text})


def _center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    """Return whether the centre of a box lies inside another box.

    Parameters
    ----------
    inner : BoundingBox
        Box whose centre is tested.
    outer : BoundingBox
        Containing box.

    Returns
    -------
    bool
        ``True`` when the centre of ``inner`` is within ``outer`` (inclusive).
    """
    center_x = (inner.x1 + inner.x2) / 2.0
    center_y = (inner.y1 + inner.y2) / 2.0
    return outer.x1 <= center_x <= outer.x2 and outer.y1 <= center_y <= outer.y2


def _covers(outer: BoundingBox, inner: BoundingBox) -> bool:
    """Return whether a box contains another box, within a 1 pt tolerance.

    Parameters
    ----------
    outer : BoundingBox
        Candidate containing box.
    inner : BoundingBox
        Box that must lie inside ``outer``.

    Returns
    -------
    bool
        ``True`` when every edge of ``inner`` is inside ``outer`` (or
        outside it by at most 1 PDF point).
    """
    tolerance = _REGION_COVER_TOLERANCE
    return (
        outer.x1 <= inner.x1 + tolerance
        and outer.y1 <= inner.y1 + tolerance
        and outer.x2 >= inner.x2 - tolerance
        and outer.y2 >= inner.y2 - tolerance
    )


def _merge_regions(
    ocr: OCRResult, indices: list[int], reading: str, bbox: BoundingBox
) -> OCRResult:
    """Replace several OCR regions by one region holding a VLM reading.

    Parameters
    ----------
    ocr : OCRResult
        Current OCR output of the page.
    indices : list of int
        Indices of the regions covered by the re-read box, in reading order.
    reading : str
        Legible VLM reading of the box.
    bbox : BoundingBox
        Re-read box, used when no region is covered.

    Returns
    -------
    OCRResult
        Copy where the covered regions are merged into one region (union
        box, minimum confidence, VLM text) placed at the first covered
        index; when nothing is covered the reading is appended as a new
        region with the page mean confidence.

    Notes
    -----
    Confidences are never raised by a re-read, so the evidence is still
    verified by the Decisions predicate.
    """
    if not indices:
        extra = OCRRegion(text=reading, bbox=bbox, confidence=ocr.mean_confidence)
        text = f"{ocr.text}\n{reading}" if ocr.text else reading
        return ocr.model_copy(update={"regions": [*ocr.regions, extra], "text": text})
    covered = [ocr.regions[index] for index in indices]
    merged = OCRRegion(
        text=reading,
        bbox=_union_bbox(covered),
        confidence=min(region.confidence for region in covered),
    )
    skipped = set(indices[1:])
    regions: list[OCRRegion] = []
    for index, region in enumerate(ocr.regions):
        if index == indices[0]:
            regions.append(merged)
        elif index not in skipped:
            regions.append(region)
    if ocr.text == "\n".join(region.text for region in ocr.regions):
        text = "\n".join(region.text for region in regions)
    else:
        text = ocr.text.replace(covered[0].text, reading, 1)
        for region in covered[1:]:
            text = text.replace(region.text, "", 1)
    return ocr.model_copy(update={"regions": regions, "text": text})


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


def _compact(text: str) -> str:
    """Lowercase a text and remove every whitespace character.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Lowercase text without whitespace (local OCR engines often drop the
        spaces between words, ``"GERENTEGENERAL:JORGE"``).
    """
    return "".join(text.lower().split())


def _quote_member_regions(
    quote: str, regions: list[OCRRegion]
) -> tuple[list[OCRRegion], float]:
    """Select the OCR lines whose text is contained in a quote, ignoring whitespace.

    Parameters
    ----------
    quote : str
        Evidence quote.
    regions : list of OCRRegion
        Page regions in reading order.

    Returns
    -------
    tuple of (list of OCRRegion, float)
        Regions of at least 6 non-space characters, not longer than the
        quote, whose compact text aligns inside the compact quote with
        ``rapidfuzz.fuzz.partial_ratio`` at 90 or above, and the fraction of
        the compact quote they cover (capped at 1.0).

    Notes
    -----
    Used when the quote cannot be aligned with the joined OCR text at all,
    which happens when the quote comes from a vision re-read and the local
    OCR engine dropped whole lines or the spaces between words.
    """
    target = _compact(quote)
    if not target:
        return [], 0.0
    members: list[OCRRegion] = []
    covered = 0
    for region in regions:
        text = _compact(region.text)
        if len(text) < _MEMBER_MIN_CHARS or len(text) > len(target):
            continue
        if fuzz.partial_ratio(text, target) >= _MEMBER_MIN_RATIO:
            members.append(region)
            covered += len(text)
    return members, min(1.0, covered / len(target))


def _widen_to_neighbours(
    box: BoundingBox, members: list[OCRRegion], regions: list[OCRRegion]
) -> BoundingBox:
    """Widen a partial evidence box to the gap between its neighbouring OCR lines.

    Parameters
    ----------
    box : BoundingBox
        Union of the OCR lines found inside the quote.
    members : list of OCRRegion
        Those lines.
    regions : list of OCRRegion
        Every region of the page.

    Returns
    -------
    BoundingBox
        Box spanning the full text width of the page, from the bottom of the
        nearest other line above to the top of the nearest other line below
        (or two median line heights beyond the box when there is none).

    Notes
    -----
    When the lines found cover less than 80% of the quote, the missing
    text was dropped by the OCR engine and lies next to them; the widened
    box keeps it inside the crop sent to the vision reader and Decisions.
    """
    others = [region for region in regions if all(region is not member for member in members)]
    heights = sorted(region.bbox.y2 - region.bbox.y1 for region in regions) or [0.0]
    step = 2.0 * heights[len(heights) // 2]
    above = [region.bbox.y2 for region in others if region.bbox.y2 <= box.y1]
    below = [region.bbox.y1 for region in others if region.bbox.y1 >= box.y2]
    return BoundingBox(
        x1=min(region.bbox.x1 for region in regions),
        y1=max(above) if above else max(0.0, box.y1 - step),
        x2=max(region.bbox.x2 for region in regions),
        y2=min(below) if below else box.y2 + step,
    )


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
        their union bounding box; failing that, the OCR lines contained in
        the quote (:func:`_quote_member_regions`), widened to their
        neighbouring lines when they cover less than 80% of the quote.
        ``None`` if nothing matches.

    Notes
    -----
    A multi-line quote spans several OCR lines; using the minimum keeps a
    single low-confidence line (for example an amount under a seal) from
    being masked by clean neighbouring lines. Regions are chosen by the
    character span of the quote inside the joined page text, found with
    ``rapidfuzz.fuzz.partial_ratio_alignment`` at 85 or above, so unrelated
    short regions elsewhere on the page do not affect the result. The
    whitespace-insensitive member fallback was added after live RapidOCR
    runs on Acta B page 2, where the engine dropped whole lines and the
    spaces between words, so quotes taken from the vision re-read could not
    be aligned and fell back to a full-page box.
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
    if matched:
        return _LocatedRegion(
            confidence=min(region.confidence for region in matched), bbox=_union_bbox(matched)
        )
    members, coverage = _quote_member_regions(evidence.source_quote, ocr.regions)
    if not members:
        return None
    box = _union_bbox(members)
    if coverage < _MEMBER_FULL_COVERAGE:
        box = _widen_to_neighbours(box, members, ocr.regions)
    return _LocatedRegion(confidence=min(region.confidence for region in members), bbox=box)


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
        context.acta_id = acta_id
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
        self._screen_injection(context)
        self._reread_low_confidence(providers.vision, context)
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
                "vlm_rereads": len(context.rereads),
                "vlm_corrections": sum(1 for item in context.rereads if item.applied),
                "suspicious_pages": sorted(context.suspicious_pages),
            },
            message=(
                f"{len(inspected)} pages inspected, "
                f"{len(inspected) - len(context.ocr_results)} native, "
                f"{len(context.ocr_results)} routed to OCR, "
                f"{len(context.rereads)} low-confidence regions re-read"
            ),
        )

        current = self._extract_and_validate(providers, spec, document_type, context, 1)
        if self._late_reread(current, spec, providers.vision, context):
            context.reextracted = True
            current = self._extract_and_validate(providers, spec, document_type, context, 2)
        self._flag_unresolved(current.resolved, spec, context)
        resolved = current.resolved
        failures = current.grounding_failures
        model_version = current.model_version
        raw_snapshot = current.raw_snapshot

        stage_start = time.perf_counter()
        escalated = self._escalate(
            resolved, providers.verifier, providers.vision, context, spec.critical_fields
        )
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
        self._apply_security_review(resolved, context)

        if not cache_hit:
            if context.vision_failures:
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="cache_not_written",
                    stage="persistence",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={"vision_failures": context.vision_failures},
                    message=f"live result for {acta_id} not cached: vision re-read failed",
                )
            elif context.verifier_failures:
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

    def _extract_and_validate(
        self,
        providers: ProviderBundle,
        spec: DocumentTypeSpec,
        document_type: str,
        context: _DocumentContext,
        pass_number: int,
    ) -> _ExtractionPass:
        """Run structured extraction, temporal resolution and validation once.

        Parameters
        ----------
        providers : ProviderBundle
            Providers selected for the document.
        spec : DocumentTypeSpec
            Document type specification.
        document_type : str
            Registered document type.
        context : _DocumentContext
            Working state with the (possibly corrected) page texts.
        pass_number : int
            ``1`` for the first pass, ``2`` for the single re-extraction.

        Returns
        -------
        _ExtractionPass
            Extraction, resolved model and validation results.
        """
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
            details={
                "model_version": model_version,
                "amendment_events": len(raw.amendment_history),
                "extraction_pass": pass_number,
            },
            message=(
                f"structured extraction completed ({model_version})"
                + (" on VLM-corrected text (re-extraction)" if pass_number > 1 else "")
            ),
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
            details={"temporal_resolution": spec.temporal_resolution, "extraction_pass": pass_number},
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
            resolved,
            page_texts=context.page_texts,
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
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
                "extraction_pass": pass_number,
            },
            message=(
                f"validation completed: {len(failures)} grounding failures, "
                f"{len(value_failures)} value-in-quote failures, "
                f"{len(temporal_failures)} temporal failures, "
                f"capital {'consistent' if consistent else 'inconsistent'}"
            ),
        )
        return _ExtractionPass(
            raw=raw,
            model_version=model_version,
            raw_snapshot=raw_snapshot,
            resolved=resolved,
            grounding_failures=failures,
            value_failures=value_failures,
            temporal_failures=temporal_failures,
            capital_consistent=consistent,
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
            context.raw_ocr_results[page.page_number] = result
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

    def _screen_injection(self, context: _DocumentContext) -> None:
        """Run the offline regex prompt-injection pre-filter on every page.

        Parameters
        ----------
        context : _DocumentContext
            Working state with the page texts (native or OCR).

        Notes
        -----
        Runs before extraction without any API call. A match raises the
        ``prompt_injection_suspected`` security flag, forces global human
        review and marks the page as suspicious, so no field whose evidence
        lies on it is auto-approved. The page text is still extracted as
        untrusted data.
        """
        for page_number in sorted(context.pages):
            matches = detect_injection_patterns(context.pages[page_number].text)
            if matches:
                self._flag_injection(
                    context,
                    page_number,
                    SECURITY_FLAG_INJECTION,
                    f"deterministic pattern match: {', '.join(matches)}",
                    {"source": "regex_prefilter", "patterns": matches},
                )

    def _flag_injection(
        self,
        context: _DocumentContext,
        page_number: int,
        flag: str,
        reason: str,
        details: dict[str, Any],
    ) -> None:
        """Raise a document-level security flag and mark the page suspicious.

        Parameters
        ----------
        context : _DocumentContext
            Working state, updated in place.
        page_number : int
            Page that raised the flag.
        flag : str
            Security flag (also the audit event name).
        reason : str
            Factual reason.
        details : dict
            Extra audit details.

        Notes
        -----
        A suspicious page prevents the auto-approval of every field whose
        evidence lies on it.
        """
        if flag not in context.security_flags:
            context.security_flags.append(flag)
        previous = context.suspicious_pages.get(page_number)
        context.suspicious_pages[page_number] = (
            reason if previous is None or reason in previous else f"{previous}; {reason}"
        )
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event=flag,
            stage="injection_screening",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            details={**details, "page_number": page_number, "reason": reason},
            message=(
                f"page {page_number}: {flag.replace('_', ' ')} ({reason}); "
                "document routed to human review, page text treated as untrusted data"
            ),
        )

    def _reread_low_confidence(
        self, vision: Optional[BaseVisionReader], context: _DocumentContext
    ) -> None:
        """Re-read low-confidence OCR regions with the vision model and patch page text.

        Parameters
        ----------
        vision : BaseVisionReader or None
            Vision reader; ``None`` skips the re-read with a WARN per page.
        context : _DocumentContext
            Working state with OCR results, updated in place.

        Notes
        -----
        Every region below ``ocr_confidence_threshold`` is cropped (bbox plus
        12 pt padding at 200 dpi) and re-read. A legible, non-empty reading
        whose normalized form differs from the OCR text replaces the region
        text in ``context.ocr_results`` and in the page text used by the
        extraction LLM; region confidences are kept so the evidence is still
        escalated to the Decisions predicate. Both readings are retained in
        ``context.rereads`` and later copied onto the evidence. A failed
        re-read keeps the OCR text and is routed to human review during
        escalation.
        """
        threshold = self._settings.ocr_confidence_threshold
        for page_number in sorted(context.ocr_results):
            ocr = context.ocr_results[page_number]
            low = [
                (index, region)
                for index, region in enumerate(ocr.regions)
                if region.confidence < threshold
            ]
            if not low:
                continue
            if vision is None:
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="vlm_reread_skipped",
                    stage="vlm_reread",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={"page_number": page_number, "low_confidence_regions": len(low)},
                    message=(
                        f"page {page_number}: {len(low)} low-confidence regions not re-read "
                        "(no vision reader configured)"
                    ),
                )
                continue
            replacements: dict[int, str] = {}
            for index, region in low:
                outcome = self._reread_region(vision, context, page_number, region)
                context.rereads.append(outcome)
                if outcome.applied and outcome.result is not None and outcome.result.reading:
                    replacements[index] = outcome.result.reading
            if replacements:
                patched = _patch_ocr_result(ocr, replacements)
                context.ocr_results[page_number] = patched
                context.pages[page_number] = context.pages[page_number].model_copy(
                    update={"text": patched.text}
                )

    def _reread_region(
        self,
        vision: BaseVisionReader,
        context: _DocumentContext,
        page_number: int,
        region: OCRRegion,
    ) -> _RegionReread:
        """Re-read one low-confidence region and log the outcome.

        Parameters
        ----------
        vision : BaseVisionReader
            Vision reader.
        context : _DocumentContext
            Working state (PDF bytes, counters, persistence log).
        page_number : int
            1-indexed page number.
        region : OCRRegion
            Low-confidence region.

        Returns
        -------
        _RegionReread
            Outcome with both readings, or the failure description.
        """
        return self._reread_bbox(
            vision,
            context,
            page_number,
            region.bbox,
            region.text,
            region.confidence,
            TRIGGER_OCR_CONFIDENCE,
        )

    def _reread_bbox(
        self,
        vision: BaseVisionReader,
        context: _DocumentContext,
        page_number: int,
        bbox: BoundingBox,
        ocr_text: str,
        ocr_confidence: float,
        trigger: str,
    ) -> _RegionReread:
        """Re-read one page box with the vision reader and log the outcome.

        Parameters
        ----------
        vision : BaseVisionReader
            Vision reader.
        context : _DocumentContext
            Working state (PDF bytes, counters, persistence log).
        page_number : int
            1-indexed page number.
        bbox : BoundingBox
            Box to re-read (a region, an evidence box or the full page).
        ocr_text : str
            Tier 1 OCR text inside the box.
        ocr_confidence : float
            Minimum OCR confidence inside the box.
        trigger : str
            Why the box is re-read.

        Returns
        -------
        _RegionReread
            Outcome with both readings, or the failure description.

        Notes
        -----
        A legible reading that differs from the OCR text is applied. For the
        missing-field trigger a non-empty reading is applied even when the
        model reports ``legible=false``: the model transcribes only visible
        characters, the page text is otherwise missing the field, and every
        evidence inside the box is still routed to review because the region
        is recorded as not fully legible.
        """
        stage_start = time.perf_counter()
        crop = self._render_crop(context, page_number, bbox)
        try:
            result = vision.reread_region(context.doc_sha256, page_number, bbox, crop, ocr_text)
        except VisionReadError as exc:
            context.vision_failures += 1
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="vlm_reread_failed",
                stage="vlm_reread",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                duration_ms=self._elapsed(stage_start),
                details={
                    "page_number": page_number,
                    "bbox": bbox.as_list(),
                    "ocr_confidence": ocr_confidence,
                    "trigger": trigger,
                    "error": exc.message,
                },
                message=f"page {page_number} region re-read failed: {exc.message}; OCR text kept",
            )
            return _RegionReread(
                page_number, bbox, ocr_text, ocr_confidence, None, exc.message, False, trigger
            )
        context.vlm_log.append(
            {
                "page_number": page_number,
                "bbox": bbox.as_list(),
                "reading": result.reading,
                "legible": result.legible,
                "engine": result.engine,
            }
        )
        differs = _normalize_space(result.reading or "") != _normalize_space(ocr_text)
        partial = trigger == TRIGGER_MISSING_FIELD and not result.legible
        applied = bool(result.reading and differs and (result.legible or partial))
        if partial and applied:
            tag, event, verdict = (
                StatusTag.WARN,
                "vlm_partial_reading_applied",
                "region not fully legible; visible transcription applied for the missing-field re-extraction",
            )
        elif not result.legible:
            tag, event, verdict = StatusTag.WARN, "vlm_region_illegible", "region not fully legible"
        elif applied:
            tag, event, verdict = StatusTag.WARN, "vlm_correction_applied", "OCR text replaced by VLM reading"
        elif differs:
            tag, event, verdict = StatusTag.WARN, "vlm_reading_empty", "VLM returned no reading; OCR text kept"
        else:
            tag, event, verdict = StatusTag.INFO, "vlm_reading_agrees", "VLM reading agrees with OCR"
        log_event(
            _LOGGER,
            tag=tag,
            event=event,
            stage="vlm_reread",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            duration_ms=self._elapsed(stage_start),
            details={
                "page_number": page_number,
                "bbox": bbox.as_list(),
                "ocr_confidence": ocr_confidence,
                "trigger": trigger,
                "legible": result.legible,
                "readings_agree": not differs,
                "engine": result.engine,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            message=(
                f"page {page_number} region {_format_bbox(bbox)} OCR "
                f"{ocr_confidence:.2f} ({trigger}): {verdict} (legible={result.legible})"
            ),
        )
        return _RegionReread(
            page_number, bbox, ocr_text, ocr_confidence, result, None, applied, trigger
        )

    def _page_box(self, context: _DocumentContext, page_number: int) -> Optional[BoundingBox]:
        """Return the full-page box of a page, memoized.

        Parameters
        ----------
        context : _DocumentContext
            Working state holding the PDF bytes.
        page_number : int
            1-indexed page number.

        Returns
        -------
        BoundingBox or None
            Page rectangle, or ``None`` when the page cannot be measured.
        """
        if page_number in context.page_boxes:
            return context.page_boxes[page_number]
        if not context.pdf_bytes:
            return None
        try:
            box = page_bbox(context.pdf_bytes, page_number)
        except PDFInspectionError:
            return None
        context.page_boxes[page_number] = box
        return box

    @staticmethod
    def _locate(context: _DocumentContext, evidence: Evidence) -> Optional[_LocatedRegion]:
        """Locate the OCR region of an evidence, preferring Tier 1 OCR geometry.

        Parameters
        ----------
        context : _DocumentContext
            Working state with the raw and corrected OCR results.
        evidence : Evidence
            Evidence on a scanned page.

        Returns
        -------
        _LocatedRegion or None
            Region found in the Tier 1 OCR output (line-level boxes), else in
            the corrected OCR output; ``None`` when neither matches.

        Notes
        -----
        A full-page re-read merges every OCR line of the page into one
        region; searching the Tier 1 lines first keeps a tight box (and a
        tight crop for the Decisions predicate) for every field whose quote
        still aligns with them.
        """
        raw = context.raw_ocr_results.get(evidence.page_number)
        corrected = context.ocr_results.get(evidence.page_number)
        if raw is not None:
            located = locate_evidence_region(raw, evidence)
            if located is not None:
                return located
        if corrected is not None and corrected is not raw:
            return locate_evidence_region(corrected, evidence)
        return None

    def _evidence_region(
        self, evidence: Evidence, context: _DocumentContext
    ) -> Optional[tuple[BoundingBox, float]]:
        """Return the box and OCR confidence used to re-read or verify an evidence.

        Parameters
        ----------
        evidence : Evidence
            Evidence on a scanned page.
        context : _DocumentContext
            Working state with OCR results.

        Returns
        -------
        tuple of (BoundingBox, float) or None
            The located OCR region (evidence box first), else the evidence
            box with its confidence, else the full page with the page
            confidence; ``None`` when nothing can be measured.
        """
        located = self._locate(context, evidence)
        if located is not None:
            return evidence.bounding_box or located.bbox, located.confidence
        if evidence.bounding_box is not None:
            return evidence.bounding_box, evidence.confidence_score
        box = self._page_box(context, evidence.page_number)
        page = context.pages.get(evidence.page_number)
        if box is None or page is None:
            return None
        return box, page.confidence_score

    def _late_targets(
        self, current: _ExtractionPass, spec: DocumentTypeSpec, context: _DocumentContext
    ) -> list[tuple[int, BoundingBox, str]]:
        """Collect the post-extraction VLM re-read targets.

        Parameters
        ----------
        current : _ExtractionPass
            First extraction pass.
        spec : DocumentTypeSpec
            Document type specification (critical and required fields).
        context : _DocumentContext
            Working state.

        Returns
        -------
        list of tuple of (int, BoundingBox, str)
            ``(page_number, bbox, trigger)`` per box, deduplicated; boxes
            overlapping a region already re-read are skipped.

        Notes
        -----
        Targets come from two triggers on scanned pages: a required
        critical field ``no_encontrado`` (every scanned page, full page)
        and a value-in-quote or capital consistency failure (the evidence
        region). Critical fields are re-read on their own region during
        escalation, after the final extraction pass.
        """
        scanned = sorted(
            number
            for number, page in context.pages.items()
            if page.source is PageSource.SCANNED_PAGE and number in context.ocr_results
        )
        if not scanned:
            return []
        model = current.resolved
        targets: list[tuple[int, BoundingBox, str]] = []

        def _add(page_number: int, bbox: BoundingBox, trigger: str) -> None:
            """Queue a target unless an existing or queued re-read covers it.

            Parameters
            ----------
            page_number : int
                1-indexed page number.
            bbox : BoundingBox
                Box to re-read.
            trigger : str
                Trigger name.

            Notes
            -----
            A full-page target is skipped only when the same box was already
            queued or re-read (earlier region re-reads do not cover the rest
            of the page); a region target is skipped when any re-read on the
            page overlaps it.
            """
            full_page = trigger == TRIGGER_MISSING_FIELD
            existing = [
                (item.page_number, item.bbox) for item in context.rereads
            ] + [(queued_page, queued_box) for queued_page, queued_box, _ in targets]
            for other_page, other_box in existing:
                if other_page != page_number:
                    continue
                if full_page and other_box.as_list() == bbox.as_list():
                    return
                if not full_page and other_box.overlaps(bbox):
                    return
            targets.append((page_number, bbox, trigger))

        missing = [
            name
            for name in spec.critical_fields
            if spec.is_required(name) and getattr(model, name).status is FieldStatus.NOT_FOUND
        ]
        if missing:
            for page_number in scanned:
                box = self._page_box(context, page_number)
                if box is not None:
                    _add(page_number, box, TRIGGER_MISSING_FIELD)
        inconsistent = list(current.value_failures)
        if not current.capital_consistent and _CAPITAL_FIELD not in inconsistent:
            inconsistent.append(_CAPITAL_FIELD)
        for name in inconsistent:
            wrapper: Optional[FieldWithEvidence[Any]] = getattr(model, name, None)
            evidence = wrapper.evidence if wrapper is not None else None
            if evidence is None or evidence.page_number not in scanned:
                continue
            region = self._evidence_region(evidence, context)
            if region is not None:
                _add(evidence.page_number, region[0], TRIGGER_VALUE_INCONSISTENCY)
        return targets

    def _late_reread(
        self,
        current: _ExtractionPass,
        spec: DocumentTypeSpec,
        vision: Optional[BaseVisionReader],
        context: _DocumentContext,
    ) -> bool:
        """Re-read post-extraction targets and decide on the single re-extraction.

        Parameters
        ----------
        current : _ExtractionPass
            First extraction pass.
        spec : DocumentTypeSpec
            Document type specification.
        vision : BaseVisionReader or None
            Vision reader; ``None`` skips the re-read with a WARN.
        context : _DocumentContext
            Working state; corrected regions are patched in place.

        Returns
        -------
        bool
            ``True`` when at least one legible reading changed the page text
            and the document has not been re-extracted yet.

        Notes
        -----
        Runs at most once per document (guarded by ``context.reextracted``),
        so a second pass can never trigger a third. Re-read outcomes are
        kept in ``context.rereads`` with their trigger; escalation later
        verifies every evidence inside them with the Decisions predicate.
        """
        if context.reextracted:
            return False
        targets = self._late_targets(current, spec, context)
        if not targets:
            return False
        if vision is None:
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="vlm_reread_skipped",
                stage="vlm_reread",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"targets": [[page, box.as_list(), trigger] for page, box, trigger in targets]},
                message=f"{len(targets)} post-extraction re-read targets skipped (no vision reader configured)",
            )
            return False
        corrected: list[str] = []
        for page_number, bbox, trigger in targets:
            ocr = context.ocr_results[page_number]
            covered = [
                index for index, region in enumerate(ocr.regions) if _center_inside(region.bbox, bbox)
            ]
            ocr_text = (
                "\n".join(ocr.regions[index].text for index in covered) if covered else ""
            )
            ocr_confidence = (
                min(ocr.regions[index].confidence for index in covered)
                if covered
                else ocr.mean_confidence
            )
            outcome = self._reread_bbox(
                vision, context, page_number, bbox, ocr_text, ocr_confidence, trigger
            )
            context.rereads.append(outcome)
            if outcome.applied and outcome.result is not None and outcome.result.reading:
                patched = _merge_regions(ocr, covered, outcome.result.reading, bbox)
                context.ocr_results[page_number] = patched
                context.pages[page_number] = context.pages[page_number].model_copy(
                    update={"text": patched.text}
                )
                corrected.append(f"page {page_number} {_format_bbox(bbox)} ({trigger})")
        if not corrected:
            return False
        log_event(
            _LOGGER,
            tag=StatusTag.WARN,
            event="reextraction_triggered",
            stage="extraction",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            details={"corrections": corrected},
            message=(
                f"{len(corrected)} region(s) corrected by the post-extraction VLM re-read; "
                "re-running extraction once on the patched text"
            ),
        )
        return True

    def _flag_unresolved(
        self, model: RawDeedExtraction, spec: DocumentTypeSpec, context: _DocumentContext
    ) -> list[str]:
        """Flag required critical fields still missing on a scanned document.

        Parameters
        ----------
        model : RawDeedExtraction
            Final resolved extraction (review flags mutated in place).
        spec : DocumentTypeSpec
            Document type specification.
        context : _DocumentContext
            Working state (scanned pages, re-reads, re-extraction flag).

        Returns
        -------
        list of str
            Names of the fields flagged.
        """
        scanned = sorted(
            number for number, page in context.pages.items() if page.source is PageSource.SCANNED_PAGE
        )
        if not scanned:
            return []
        reread_pages = sorted(
            {item.page_number for item in context.rereads if item.trigger == TRIGGER_MISSING_FIELD}
        )
        flagged: list[str] = []
        for name in spec.critical_fields:
            wrapper: FieldWithEvidence[Any] = getattr(model, name)
            if not spec.is_required(name) or wrapper.status is not FieldStatus.NOT_FOUND:
                continue
            if reread_pages:
                reason = (
                    f"Required field {name} not found after VLM re-read of scanned page(s) "
                    f"{', '.join(str(page) for page in reread_pages)}"
                    + (
                        " and one re-extraction"
                        if context.reextracted
                        else "; the re-read did not change the page text"
                    )
                )
            else:
                reason = (
                    f"Required field {name} not found on a document with scanned page(s) "
                    f"{', '.join(str(page) for page in scanned)}; VLM re-read not available"
                )
            wrapper.flag_for_review(reason)
            flagged.append(name)
            log_event(
                _LOGGER,
                tag=StatusTag.REVIEW_REQUIRED,
                event="critical_field_unresolved",
                stage="validation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"field": name, "scanned_pages": scanned, "reextracted": context.reextracted},
                message=f"{name}: {reason}",
            )
        return flagged

    def _apply_security_review(
        self, model: RawDeedExtraction, context: _DocumentContext
    ) -> list[str]:
        """Prevent auto-approval of fields whose evidence is on a suspicious page.

        Parameters
        ----------
        model : RawDeedExtraction
            Final extraction (review flags mutated in place).
        context : _DocumentContext
            Working state with suspicious pages.

        Returns
        -------
        list of str
            Names of the fields flagged.
        """
        if not context.suspicious_pages:
            return []
        flagged: list[str] = []
        for ref in self._evidence_entries(model):
            page_number = ref.evidence.page_number
            if page_number not in context.suspicious_pages:
                continue
            reason = (
                f"Prompt injection suspected on page {page_number} "
                f"({context.suspicious_pages[page_number]}); field not auto-approved"
            )
            targets = (ref.wrapper,) if ref.wrapper is not None else ref.dependents
            for target in targets:
                target.flag_for_review(reason)
            if ref.name not in flagged:
                flagged.append(ref.name)
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="security_review_applied",
            stage="validation",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            details={"pages": sorted(context.suspicious_pages), "fields": flagged},
            message=(
                f"fields with evidence on suspicious page(s) "
                f"{', '.join(str(page) for page in sorted(context.suspicious_pages))} "
                f"routed to review: {', '.join(flagged) or 'none'}"
            ),
        )
        return flagged

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
        live LLM output cannot self-assert confidence. Regions are located
        with :meth:`_locate` (Tier 1 OCR geometry first), so a full-page
        re-read never widens the box of a field whose quote is found on a
        single OCR line.
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
            located = self._locate(context, evidence)
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
            A value-based predicate for the field (or list item) the evidence
            supports; a substance-of-passage predicate as the fallback.

        Notes
        -----
        Live runs showed that asking whether a crop reads a quote verbatim
        fails on clean scans (quotes longer than the crop, ISO dates versus
        dates in words, accents), so predicates state what the crop shows about
        the extracted value, with no page numbers or terms absent from the crop.
        """
        value = wrapper.value if wrapper is not None else None
        if name == _CAPITAL_FIELD and value is not None:
            return f"The image shows the capital social amount ${float(value.monto_numerico):,.2f} MXN"
        if name == "fecha_constitucion" and isinstance(value, str):
            return f"The image shows the date {_date_in_words(value)}"
        if name == "numero_escritura" and isinstance(value, str):
            return f"The image shows the instrument or deed number {value}"
        if name == "razon_social" and isinstance(value, str):
            return f'The image shows the company name "{value}"'
        if name == "notario" and value is not None:
            return (
                f"The image shows the notary {value.nombre} and notary office number "
                f"{value.numero_notaria}"
            )
        item = next(
            (entry for entry in value if getattr(entry, "evidence", None) is evidence),
            None,
        ) if isinstance(value, list) else None
        if name == "socios" and item is not None:
            return (
                f"The image shows the shareholder {item.nombre} with {item.acciones} shares "
                f"or social parts ({item.porcentaje:g}%)"
            )
        if name == "representantes_legales" and item is not None:
            return f"The image shows {item.nombre} appointed as {item.cargo}"
        if name == "poderes" and item is not None:
            return f"The image shows {item.otorgado_a} granted a {item.tipo_poder}"
        quote = " ".join(evidence.source_quote.split())[:_PREDICATE_QUOTE_LIMIT]
        return (
            f"The image shows the substance of this passage "
            f'(ignore accents, punctuation and line breaks): "{quote}"'
        )

    def _escalate(
        self,
        model: RawDeedExtraction,
        verifier: BaseVerifierProvider,
        vision: Optional[BaseVisionReader],
        context: _DocumentContext,
        critical_fields: tuple[str, ...] = (),
    ) -> int:
        """Re-read and verify scanned evidence, then route each field to review.

        Parameters
        ----------
        model : RawDeedExtraction
            Resolved extraction (mutated in place).
        verifier : BaseVerifierProvider
            Tier 2 Decisions predicate verifier.
        vision : BaseVisionReader or None
            Vision reader for critical-field regions; ``None`` skips the
            re-read with a WARN (the predicate still runs).
        context : _DocumentContext
            Working state with pages, OCR results and verifier memo.
        critical_fields : tuple of str, default ()
            Fields always re-read and verified on scanned pages.

        Returns
        -------
        int
            Number of distinct predicates sent to the verifier.

        Notes
        -----
        Only evidence on scanned pages is considered; native pages never
        reach the vision model or Decisions. An evidence is escalated when
        it belongs to a critical field (always), when its OCR region has
        confidence below ``ocr_confidence_threshold`` (Tier 1), or when it
        lies inside a post-extraction re-read box. A critical field whose
        own region was not re-read yet is re-read now
        (``TRIGGER_CRITICAL_FIELD``). The Decisions predicate then runs on
        a tight crop of the field region; ``|2p - 1|`` replaces the
        evidence confidence and :meth:`_decide_review` applies the
        per-field review rules.
        """
        verified_count = 0
        ocr_threshold = self._settings.ocr_confidence_threshold
        for ref in self._evidence_entries(model):
            name, wrapper, evidence = ref.name, ref.wrapper, ref.evidence
            page = context.pages.get(evidence.page_number)
            ocr = context.ocr_results.get(evidence.page_number)
            if page is None or ocr is None or page.source is not PageSource.SCANNED_PAGE:
                continue
            located = self._locate(context, evidence)
            low_ocr = located is not None and located.confidence < ocr_threshold
            critical = wrapper is not None and name in critical_fields
            guess = evidence.bounding_box or (located.bbox if located is not None else None)
            late = [
                item
                for item in context.rereads
                if item.trigger not in (TRIGGER_OCR_CONFIDENCE, TRIGGER_CRITICAL_FIELD)
                and item.page_number == evidence.page_number
                and guess is not None
                and item.bbox.overlaps(guess)
            ]
            if not low_ocr and not critical and not late:
                continue
            if located is not None:
                bbox = evidence.bounding_box or located.bbox
                ocr_confidence = located.confidence
            else:
                region = self._evidence_region(evidence, context)
                if region is None:
                    self._route_unlocated(ref, context)
                    continue
                bbox, ocr_confidence = region
            trigger_note: Optional[str] = None
            if not low_ocr:
                trigger_note = (
                    f"critical field {name} on scanned page {evidence.page_number}"
                    if critical
                    else "post-extraction VLM re-read ("
                    + ", ".join(sorted({item.trigger for item in late}))
                    + ")"
                )
            if critical:
                self._ensure_region_reread(vision, context, name, evidence.page_number, bbox)
            predicate = self._build_predicate(name, wrapper, evidence)
            key = (evidence.page_number, tuple(bbox.as_list()), predicate)
            rereads = self._own_rereads(context, evidence.page_number, bbox)
            self._attach_rereads(evidence, rereads)
            log_event(
                _LOGGER,
                tag=StatusTag.WARN if low_ocr else StatusTag.INFO,
                event=(
                    "tier1_escalation"
                    if low_ocr
                    else "critical_field_escalation"
                    if critical
                    else "reread_escalation"
                ),
                stage="escalation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={
                    "field": name,
                    "page_number": evidence.page_number,
                    "ocr_confidence": ocr_confidence,
                    "ocr_threshold": ocr_threshold,
                    "bbox": bbox.as_list(),
                    "vlm_rereads": len(rereads),
                    "trigger": "ocr_confidence" if low_ocr else trigger_note,
                },
                message=(
                    f"{name}: OCR confidence {ocr_confidence:.2f} below "
                    f"{ocr_threshold:.2f} on page {evidence.page_number}; "
                    "escalating to predicate verification"
                    if low_ocr
                    else f"{name}: OCR confidence {ocr_confidence:.2f} on page "
                    f"{evidence.page_number}; {trigger_note}; escalating to predicate verification"
                ),
            )
            crop = self._render_crop(context, evidence.page_number, bbox)
            if crop is not None:
                evidence.crop_path = self._persist_crop(context, name, evidence.page_number, crop)
            result = context.verified.get(key)
            if result is None:
                try:
                    result = verifier.verify_predicate(
                        context.doc_sha256,
                        evidence.page_number,
                        predicate,
                        # Region text only: the whole OCR page measurably degraded live answers
                        evidence.ocr_reading or "",
                        bbox,
                        image_png=crop,
                    )
                except VerifierError as exc:
                    self._route_verifier_failure(
                        ref, ocr_confidence, bbox, exc, context, trigger_note
                    )
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
            evidence.verification_probability = result.probability
            if evidence.grounding_verified:
                evidence.confidence_score = result.calibrated_confidence
            else:
                evidence.confidence_score = min(
                    evidence.confidence_score, result.calibrated_confidence
                )
            evidence.confidence_source = ConfidenceSource.DECISIONS_PREDICATE
            self._decide_review(
                ref, ocr_confidence, bbox, result, rereads, context, trigger_note
            )
        return verified_count

    @classmethod
    def _own_rereads(
        cls, context: _DocumentContext, page_number: int, bbox: BoundingBox
    ) -> list[_RegionReread]:
        """Return the re-reads that are facts about one field region.

        Parameters
        ----------
        context : _DocumentContext
            Working state with every re-read.
        page_number : int
            1-indexed page of the field evidence.
        bbox : BoundingBox
            Field region (also the predicate crop box).

        Returns
        -------
        list of _RegionReread
            Region re-reads on the page that cover the field region, or that
            lie inside it (an OCR line of the field quote), in order.

        Notes
        -----
        Shared full-page re-reads of the missing-field path are excluded:
        their ``legible`` verdict describes the whole page, not the field.
        A page-level re-read made for another field is excluded too, unless
        the field's own box is page-level as well.
        """
        page_level_field = cls._is_page_level(context, page_number, bbox)
        return [
            item
            for item in context.rereads
            if item.page_number == page_number
            and item.trigger != TRIGGER_MISSING_FIELD
            and (_covers(item.bbox, bbox) or _center_inside(item.bbox, bbox))
            and (page_level_field or not cls._is_page_level(context, page_number, item.bbox))
        ]

    @staticmethod
    def _is_page_level(context: _DocumentContext, page_number: int, bbox: BoundingBox) -> bool:
        """Return whether a box covers the whole text area of a scanned page.

        Parameters
        ----------
        context : _DocumentContext
            Working state with the Tier 1 OCR output.
        page_number : int
            1-indexed page number.
        bbox : BoundingBox
            Re-read box.

        Returns
        -------
        bool
            ``True`` when the box covers at least 90% of the area of the
            union of the Tier 1 OCR lines of the page.

        Notes
        -----
        The ``legible`` verdict of such a re-read describes the page (for
        example a seal elsewhere on it), not one field, so it never flags a
        field by itself; the field's own Decisions predicate decides.
        """
        ocr = context.raw_ocr_results.get(page_number) or context.ocr_results.get(page_number)
        if ocr is None or not ocr.regions:
            return False
        text = _union_bbox(list(ocr.regions))
        width = min(bbox.x2, text.x2) - max(bbox.x1, text.x1)
        height = min(bbox.y2, text.y2) - max(bbox.y1, text.y1)
        area = (text.x2 - text.x1) * (text.y2 - text.y1)
        if width <= 0 or height <= 0 or area <= 0:
            return False
        return width * height / area >= _PAGE_LEVEL_TEXT_COVERAGE

    def _ensure_region_reread(
        self,
        vision: Optional[BaseVisionReader],
        context: _DocumentContext,
        name: str,
        page_number: int,
        bbox: BoundingBox,
    ) -> None:
        """Re-read a critical field region unless a region re-read already covers it.

        Parameters
        ----------
        vision : BaseVisionReader or None
            Vision reader; ``None`` logs a WARN and skips the re-read.
        context : _DocumentContext
            Working state (re-reads appended in place).
        name : str
            Contract field name (audit records).
        page_number : int
            1-indexed page number.
        bbox : BoundingBox
            Field region.

        Notes
        -----
        The reading is recorded for the review decision and the audit trail
        but does not patch the page text: extraction has already run.
        """
        page_level_field = self._is_page_level(context, page_number, bbox)
        if any(
            item.page_number == page_number
            and item.trigger != TRIGGER_MISSING_FIELD
            and _covers(item.bbox, bbox)
            and (page_level_field or not self._is_page_level(context, page_number, item.bbox))
            for item in context.rereads
        ):
            return
        if vision is None:
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="vlm_reread_skipped",
                stage="vlm_reread",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"field": name, "page_number": page_number, "bbox": bbox.as_list()},
                message=(
                    f"{name}: critical field region on page {page_number} not re-read "
                    "(no vision reader configured); Decisions predicate only"
                ),
            )
            return
        ocr = context.ocr_results[page_number]
        covered = [region for region in ocr.regions if _center_inside(region.bbox, bbox)]
        ocr_text = "\n".join(region.text for region in covered)
        ocr_confidence = (
            min(region.confidence for region in covered) if covered else ocr.mean_confidence
        )
        context.rereads.append(
            self._reread_bbox(
                vision, context, page_number, bbox, ocr_text, ocr_confidence, TRIGGER_CRITICAL_FIELD
            )
        )

    def _route_unlocated(self, ref: _EvidenceRef, context: _DocumentContext) -> None:
        """Flag a field whose scanned evidence region cannot be measured.

        Parameters
        ----------
        ref : _EvidenceRef
            Evidence and its dependent fields.
        context : _DocumentContext
            Working state (job and hash).

        Notes
        -----
        Happens only when neither the OCR regions, the evidence box nor the
        page rectangle are available; the field is never auto-approved.
        """
        reason = (
            f"Evidence region on scanned page {ref.evidence.page_number} could not be located; "
            "VLM re-read and Decisions predicate not run"
        )
        for dependent in ref.dependents:
            dependent.flag_for_review(reason)
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="evidence_region_unavailable",
            stage="escalation",
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            details={"field": ref.name, "page_number": ref.evidence.page_number},
            message=f"{ref.name}: {reason}",
        )

    @staticmethod
    def _attach_rereads(evidence: Evidence, rereads: list[_RegionReread]) -> None:
        """Copy the OCR and VLM readings of re-read regions onto an evidence.

        Parameters
        ----------
        evidence : Evidence
            Evidence inside one or more re-read regions (mutated in place).
        rereads : list of _RegionReread
            Re-read regions overlapping the evidence box, in reading order.

        Notes
        -----
        Several regions are joined with single spaces; ``vlm_legible`` is
        ``True`` only when every region was re-read and judged legible.
        """
        if not rereads:
            return
        evidence.ocr_reading = " ".join(item.ocr_text for item in rereads)
        readings = [
            item.result.reading for item in rereads if item.result is not None and item.result.reading
        ]
        evidence.vlm_reading = " ".join(readings) if readings else None
        evidence.vlm_legible = all(
            item.result is not None and item.result.legible for item in rereads
        )

    def _decide_review(
        self,
        ref: _EvidenceRef,
        ocr_confidence: float,
        bbox: BoundingBox,
        result: VerifierResult,
        rereads: list[_RegionReread],
        context: _DocumentContext,
        trigger_note: Optional[str] = None,
    ) -> None:
        """Apply the human-review rules after the Decisions predicate.

        Parameters
        ----------
        ref : _EvidenceRef
            Evidence and its dependent fields.
        ocr_confidence : float
            Tier 1 OCR confidence of the located region.
        bbox : BoundingBox
            Region of interest (also the crop box).
        result : VerifierResult
            Decisions predicate result.
        rereads : list of _RegionReread
            Re-reads of the field's own region (:meth:`_own_rereads`);
            shared full-page re-reads are never included.
        context : _DocumentContext
            Working state (job and hash).
        trigger_note : str or None, default None
            Why an evidence with OCR confidence at or above the threshold was
            escalated (for example a critical field on a scanned page);
            ``None`` means the OCR confidence itself was below the threshold.

        Notes
        -----
        Review is decided per field from facts about its own value: the
        calibrated confidence is below ``verifier_confidence_threshold``,
        the VLM judged the field's own region not fully legible and Decisions
        did not confirm the value either, or the VLM re-read of that region
        failed. Decisions has the final visual word: a ``legible=false`` own
        region whose value Decisions confirms at or above the threshold is not
        flagged, because live runs showed the VLM marks cleanly scanned crops
        illegible when a line is cut at the crop edge. A ``legible=false`` verdict on a box
        that covers the whole text area of the page (:meth:`_is_page_level`)
        is not a fact about one field and never flags it by itself. A VLM correction that
        disagrees with OCR is accepted without review only when Decisions
        confirms it at or above the threshold (``[OK] vlm_correction_confirmed``).
        When low OCR confidence and low Decisions confidence are the only
        triggers the reason keeps the spec-02 wording exactly; an evidence
        escalated for another reason states that reason instead.
        """
        evidence = ref.evidence
        page_number = evidence.page_number
        threshold = self._settings.verifier_confidence_threshold
        low_decisions = result.calibrated_confidence < threshold
        corrected = any(item.applied for item in rereads)
        illegible = [
            item
            for item in rereads
            if item.result is not None
            and not item.result.legible
            and not self._is_page_level(context, item.page_number, item.bbox)
        ]
        failed = [item for item in rereads if item.error is not None]
        facts = (
            f"OCR confidence {ocr_confidence:.2f}; Decisions p={result.probability:.2f}, "
            f"calibrated {result.calibrated_confidence:.2f} vs {threshold:.2f}; page {page_number} "
            f"bbox {_format_bbox(bbox)}."
        )
        parts: list[str] = []
        if low_decisions and trigger_note is None:
            parts.append(
                f"Low OCR confidence ({ocr_confidence:.2f}) and Decisions predicate "
                f"confidence ({result.calibrated_confidence:.2f} < {threshold:.2f}) due to "
                f"overlapping notary seal on page {page_number} bbox {_format_bbox(bbox)}."
            )
        elif low_decisions:
            parts.append(
                f"Escalated despite OCR confidence {ocr_confidence:.2f} because of {trigger_note}; "
                f"Decisions predicate confidence ({result.calibrated_confidence:.2f} < "
                f"{threshold:.2f}) on page {page_number} bbox {_format_bbox(bbox)}."
            )
        if low_decisions:
            if corrected:
                parts.append(
                    "OCR and VLM readings disagree (OCR "
                    + _summarize_reading(evidence.ocr_reading)
                    + ", VLM "
                    + _summarize_reading(evidence.vlm_reading)
                    + ") and Decisions did not confirm the VLM correction."
                )
        # An empty VLM reading always counts; a non-empty illegible reading needs low Decisions too
        unread = any(not (item.result.reading or "").strip() for item in illegible if item.result)
        if illegible and (low_decisions or unread):
            parts.append(
                "VLM re-read judged the region not fully legible (legible=false, reading "
                + _summarize_reading(" ".join(item.result.reading or "" for item in illegible if item.result) or None)
                + "); "
                + facts
            )
        if failed:
            parts.append(
                f"VLM re-read unavailable ({'; '.join(item.error or '' for item in failed)}); {facts}"
            )
        if not parts:
            if corrected:
                log_event(
                    _LOGGER,
                    tag=StatusTag.OK,
                    event="vlm_correction_confirmed",
                    stage="escalation",
                    job_id=context.job_id,
                    doc_sha256=context.doc_sha256,
                    details={
                        "field": ref.name,
                        "page_number": page_number,
                        "ocr_confidence": ocr_confidence,
                        "probability": result.probability,
                        "calibrated_confidence": result.calibrated_confidence,
                        "bbox": bbox.as_list(),
                    },
                    message=(
                        f"{ref.name}: VLM correction confirmed by Decisions "
                        f"(p={result.probability:.2f}, calibrated "
                        f"{result.calibrated_confidence:.2f} >= {threshold:.2f})"
                    ),
                )
            return
        reason = " ".join(parts)
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
                "field": ref.name,
                "page_number": page_number,
                "ocr_confidence": ocr_confidence,
                "probability": result.probability,
                "calibrated_confidence": result.calibrated_confidence,
                "verifier_threshold": threshold,
                "bbox": bbox.as_list(),
                "vlm_legible": evidence.vlm_legible,
                "vlm_corrected": corrected,
                "vlm_failed": bool(failed),
            },
            message=(
                f"{ref.name}: OCR {ocr_confidence:.2f}, p={result.probability:.2f}, "
                f"calibrated {result.calibrated_confidence:.2f} vs {threshold:.2f}, "
                f"VLM legible={evidence.vlm_legible}"
            ),
        )

    def _persist_crop(
        self, context: _DocumentContext, name: str, page_number: int, crop: bytes
    ) -> Optional[str]:
        """Write a verification crop under ``outputs/crops``.

        Parameters
        ----------
        context : _DocumentContext
            Working state (acta identifier and names already used).
        name : str
            Field or event name (``amendment_history[i]`` becomes
            ``amendment_history_i``).
        page_number : int
            1-indexed page number.
        crop : bytes
            PNG crop.

        Returns
        -------
        str or None
            Project-relative path, or ``None`` (with a WARN record) when the
            crop cannot be written.

        Notes
        -----
        Files are named ``{acta_id}_p{page}_{field}_bbox.png``; a second crop
        for the same field and page gets a ``_2``, ``_3`` ... suffix.
        """
        slug = "".join(char if char.isalnum() or char == "_" else "_" for char in name).strip("_")
        base = f"{context.acta_id}_p{page_number}_{slug}_bbox"
        crop_name = f"{base}.png"
        counter = 2
        while crop_name in context.crop_names:
            crop_name = f"{base}_{counter}.png"
            counter += 1
        context.crop_names.add(crop_name)
        try:
            return self._repository.save_crop(crop_name, crop)
        except StorageError as exc:
            log_event(
                _LOGGER,
                tag=StatusTag.WARN,
                event="crop_not_written",
                stage="escalation",
                job_id=context.job_id,
                doc_sha256=context.doc_sha256,
                details={"crop_name": crop_name, "error": exc.message},
                message=f"cannot persist crop {crop_name}: {exc.message}",
            )
            return None

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
        trigger_note: Optional[str] = None,
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
        trigger_note : str or None, default None
            Escalation reason when the OCR confidence was not low.

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
        cause = (
            f"Low OCR confidence ({ocr_confidence:.2f})"
            if trigger_note is None
            else f"Escalation due to {trigger_note} (OCR confidence {ocr_confidence:.2f})"
        )
        reason = (
            f"{cause} and Decisions predicate verification "
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
            Working state with OCR results, VLM re-reads and verifier answers.
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
                for number, result in sorted(
                    (context.raw_ocr_results or context.ocr_results).items()
                )
            },
            "vlm_reads": context.vlm_log,
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
        security_flags = list(context.security_flags)
        global_review = bool(review_fields) or bool(security_flags)
        metadata = AuditMetadata(
            job_id=context.job_id,
            doc_sha256=context.doc_sha256,
            document_filename=source_name,
            model_version=model_version,
            total_pages=len(context.pages),
            native_pages_count=native_count,
            ocr_pages_count=len(context.pages) - native_count,
            global_human_review_required=global_review,
            processing_time_ms=self._elapsed(started),
            security_flags=security_flags,
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
            tag=StatusTag.REVIEW_REQUIRED if global_review else StatusTag.OK,
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
                "security_flags": security_flags,
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
                + (f"; security flags: {', '.join(security_flags)}" if security_flags else "")
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
            security_flags=security_flags,
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
        use OpenAI providers (extraction, vision re-read and the Decisions
        predicate) with RapidOCR for scanned pages.

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
        vision=DeterministicVLMProvider(repository),
    )
    registry = DocumentRegistry.default()
    live_bundle: Optional[ProviderBundle] = None
    if live and settings.live_mode_available:
        live_bundle = ProviderBundle(
            ocr=LazyOCRProvider(repository, "rapidocr"),
            verifier=build_verifier_provider(settings, repository, True),
            llm=build_llm_provider(settings, repository, True, registry),
            vision=build_vision_provider(settings, repository, True),
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
