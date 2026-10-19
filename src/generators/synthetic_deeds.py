"""Synthetic deed PDF generator with synchronized cache and ground truth.

Renders ``acta_a.pdf``, ``acta_b.pdf`` and ``acta_c.pdf`` from the
content in :mod:`src.generators.deed_content`, hashes each binary, writes
the pre-verified raw extraction to ``data/cache/{sha256}.json`` and writes
``data/ground_truth/ground_truth.json``. Self-checks confirm that every
cached ``source_quote`` is grounded in the text of its page.

Notes
-----
Determinism
    ReportLab runs with ``invariant=1`` (fixed creation date and document
    ID) and the scanned page image is drawn with a fixed canvas size, a
    font bundled with Pillow and seeded noise, so repeated runs with the
    same locked dependencies produce byte-identical PDFs and hashes.

Page geometry
    US Letter, 612 x 792 PDF points. The scanned image for Acta B page 2
    is 1700 x 2200 pixels (8.5 x 11 in at 200 dpi) and covers the whole
    page, so one PDF point equals 200 / 72 image pixels. Region bounding
    boxes are in PDF points with a top-left origin (PyMuPDF convention);
    the capital clause is drawn inside ``[72.0, 410.0, 520.0, 455.0]``.

Text layout
    Native paragraphs are wrapped only at spaces (never hyphenated) and
    drawn left-aligned in Helvetica (Latin-1 covers Ñ, É, Ó). Downstream
    grounding normalizes whitespace, so wrapped quotes still match.

Cache file format (``data/cache/{sha256}.json``)
    ``{"doc_sha256", "document_filename", "acta_id", "document_type",
    "model_version": "pre-verified-cache-v1", "raw_extraction",
    "ocr_pages": {"<page>": OCRResult}, "vlm_reads": [...],
    "verifier_responses": [...], "decision_scores": [...]}``. ``vlm_reads``
    holds one ``{page_number, bbox, reading, legible, engine}`` entry per OCR
    region below 0.85 and per clean critical region on a low-legibility page
    (the simulated vision re-read). ``decision_scores`` holds simulated
    Decisions ``score`` answers ``{page_number, name, outcome, score,
    confidence, probabilities, engine}``: ``PROMPT_INJECTION_RISK_SCORE`` for
    every page and ``OCR_LEGIBILITY_SCORE`` for every scanned page.
    ``raw_extraction`` is a validated :class:`RawDeedExtraction` dumped
    with ``model_dump(by_alias=True, mode="json")``.

Ground truth format (``data/ground_truth/ground_truth.json``)
    A JSON object keyed by ``acta_id``. Each value has::

        {
          "acta_id": "acta_c",
          "document_filename": "acta_c.pdf",
          "doc_sha256": "<hex>",
          "total_pages": 4,
          "scenario": "<English description>",
          "native_pages": [1, 2, 3, 4],
          "scanned_pages": [],
          "fields": {
            "razon_social":       {"value": str, "page": int},
            "fecha_constitucion": {"value": "YYYY-MM-DD", "page": int},
            "numero_escritura":   {"value": "30,100", "page": int},
            "notario": {"value": {"nombre", "numero_notaria",
                                  "entidad_federativa"}, "page": int},
            "socios": {"value": [{"nombre", "rfc", "acciones",
                                  "porcentaje", "page"}], "page": int},
            "representantes_legales": {"value": [{"nombre",
                "nombre_normalizado", "cargo", "fecha_nombramiento",
                "page"}], "page": int},
            "poderes": {"value": [{"otorgado_a" (normalized name),
                "tipo_poder", "modalidad", "page"}], "page": int},
            "capital_social_actual": {"value": {"monto_numerico",
                "moneda", "monto_en_letra", "fecha_acto_vigente",
                "escritura_origen"}, "page": int}
          },
          "revoked_representatives": [normalized names],
          "expected_human_review_fields": [field names],
          "amendment_history": {"event_count": int,
              "event_types": [AmendmentType values],
              "events": [{"event_type", "fecha_acto", "numero_escritura",
                          "page", "sujeto_afectado", "monto_nuevo"}]}
        }

    Lists hold the ACTIVE state after temporal resolution (revoked people
    and their powers are excluded). The field-level ``page`` is the
    expected evidence page of the field (first item for lists).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from src.core.config import Settings
from src.core.exceptions import PipelineError
from src.core.logger import StatusTag, get_logger, log_event
from src.generators.deed_content import (
    CACHE_MODEL_VERSION,
    DEED_SPECS,
    DOCUMENT_TYPE,
    RAW_EXTRACTION_BUILDERS,
    BlockStyle,
    DeedSpec,
    NativePageSpec,
    RasterPageSpec,
    TextBlock,
    build_decision_scores,
    build_ground_truth_fields,
    build_ocr_result,
    build_verifier_responses,
    build_vlm_reads,
)
from src.schemas.deed_models import Evidence, FieldWithEvidence, RawDeedExtraction

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN_X: float = 72.0
CONTENT_WIDTH: float = PAGE_WIDTH - 2 * MARGIN_X
BODY_TOP: float = 84.0
BODY_BOTTOM_LIMIT: float = 738.0

RASTER_DPI: int = 200
RASTER_WIDTH_PX: int = 1700
RASTER_HEIGHT_PX: int = 2200
PX_PER_POINT: float = RASTER_DPI / 72.0
RASTER_FONT_PX: int = 30
RASTER_LINE_PX: int = 40
RASTER_SEAL_FONT_PX: int = 34
SMUDGE_INK: tuple[int, int, int] = (52, 34, 118)
SMUDGE_CORE_ALPHA: int = 238
SMUDGE_HALO_ALPHA: int = 150
SMUDGE_BLOTS_PER_FRAGMENT: int = 26

_STAGE: str = "generation"

_STYLE_FONTS: dict[BlockStyle, tuple[str, float, float, float, float]] = {
    BlockStyle.TITLE: ("Helvetica-Bold", 11.0, 15.0, 4.0, 10.0),
    BlockStyle.HEADING: ("Helvetica-Bold", 10.5, 14.0, 6.0, 6.0),
    BlockStyle.BODY: ("Helvetica", 10.5, 14.5, 0.0, 8.0),
    BlockStyle.SIGNATURE: ("Helvetica", 9.5, 13.0, 0.0, 2.0),
}
"""Style -> (font name, size, leading, space before, space after)."""


@dataclass(frozen=True)
class GeneratedDeed:
    """Result of generating one synthetic deed.

    Parameters
    ----------
    acta_id : str
        Deed identifier.
    pdf_path : Path
        Written PDF path.
    sha256 : str
        Hex SHA-256 of the PDF bytes.
    cache_path : Path
        Written cache payload path.
    total_pages : int
        Number of pages.
    """

    acta_id: str
    pdf_path: Path
    sha256: str
    cache_path: Path
    total_pages: int


# Section: text helpers


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into single spaces.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Stripped text with single spaces.
    """
    return " ".join(text.split())


def count_clean_chars(text: str) -> int:
    """Count alphanumeric characters (``str.isalnum``).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Number of alphanumeric characters.
    """
    return sum(1 for char in text if char.isalnum())


def _wrap_words(text: str, max_width: float, measure: Any) -> list[str]:
    """Greedily wrap text at spaces without splitting words.

    Parameters
    ----------
    text : str
        Text to wrap.
    max_width : float
        Maximum line width in the unit returned by ``measure``.
    measure : callable
        Function returning the width of a string.

    Returns
    -------
    list of str
        Wrapped lines.

    Raises
    ------
    PipelineError
        If a single word is wider than ``max_width``.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        if measure(word) > max_width:
            raise PipelineError("word wider than the text column", {"word": word})
        candidate = f"{current} {word}" if current else word
        if measure(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# Section: native page rendering


def _draw_native_page(canvas: Canvas, page: NativePageSpec) -> None:
    """Draw one native page with running header, body and footer.

    Parameters
    ----------
    canvas : Canvas
        ReportLab canvas positioned on a fresh page.
    page : NativePageSpec
        Page content.

    Raises
    ------
    PipelineError
        If the body overflows the printable area.
    """
    canvas.setLineWidth(0.6)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_X, PAGE_HEIGHT - 44.0, page.running_header)
    canvas.line(MARGIN_X, PAGE_HEIGHT - 50.0, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 50.0)
    canvas.line(MARGIN_X, PAGE_HEIGHT - 748.0, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 748.0)
    canvas.rect(MARGIN_X - 18.0, 30.0, PAGE_WIDTH - 2 * (MARGIN_X - 18.0), PAGE_HEIGHT - 60.0)

    cursor = BODY_TOP
    previous: TextBlock | None = None
    for block in page.blocks:
        font, size, leading, space_before, space_after = _STYLE_FONTS[block.style]
        if block.style is BlockStyle.SIGNATURE and (
            previous is None or previous.style is not BlockStyle.SIGNATURE
        ):
            cursor += 42.0
            canvas.line(
                PAGE_WIDTH / 2 - 120.0,
                PAGE_HEIGHT - cursor + 4.0,
                PAGE_WIDTH / 2 + 120.0,
                PAGE_HEIGHT - cursor + 4.0,
            )
            cursor += 8.0
        cursor += space_before
        lines = _wrap_words(
            block.text, CONTENT_WIDTH, lambda value, f=font, s=size: stringWidth(value, f, s)
        )
        canvas.setFont(font, size)
        for line in lines:
            cursor += leading
            baseline = PAGE_HEIGHT - cursor
            if block.style in (BlockStyle.TITLE, BlockStyle.HEADING, BlockStyle.SIGNATURE):
                canvas.drawCentredString(PAGE_WIDTH / 2, baseline, line)
            else:
                canvas.drawString(MARGIN_X, baseline, line)
        cursor += space_after
        previous = block
    if cursor > BODY_BOTTOM_LIMIT:
        raise PipelineError("native page body overflows the printable area", {"cursor": cursor})
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN_X, PAGE_HEIGHT - 760.0, page.footer)


# Section: raster page rendering


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the font bundled with Pillow at a given pixel size.

    Parameters
    ----------
    size : int
        Font size in pixels.

    Returns
    -------
    FreeTypeFont or ImageFont
        Scalable bundled font, or the fixed bitmap default when FreeType
        support is unavailable.
    """
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, ImportError, OSError):
        return ImageFont.load_default()


def _to_px(value: float) -> float:
    """Convert PDF points to image pixels.

    Parameters
    ----------
    value : float
        Length in PDF points.

    Returns
    -------
    float
        Length in pixels at 200 dpi.
    """
    return value * PX_PER_POINT


def _draw_seal(image: Image.Image, page: RasterPageSpec) -> Image.Image:
    """Overlay a semi-transparent rotated notarial seal.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB page image.
    page : RasterPageSpec
        Raster page specification (seal text and target region).

    Returns
    -------
    PIL.Image.Image
        New RGB image with the seal composited.
    """
    seal_w, seal_h = 980, 300
    seal = Image.new("RGBA", (seal_w, seal_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(seal)
    ink_fill = (72, 52, 150, 105)
    ink_line = (64, 44, 140, 190)
    draw.ellipse((6, 6, seal_w - 6, seal_h - 6), fill=ink_fill, outline=ink_line, width=9)
    draw.ellipse((34, 30, seal_w - 34, seal_h - 30), outline=ink_line, width=4)
    font = _load_font(RASTER_SEAL_FONT_PX)
    text_w = draw.textlength(page.seal_text, font=font)
    draw.text(((seal_w - text_w) / 2, seal_h / 2 - 22), page.seal_text, font=font, fill=(60, 40, 135, 215))
    for index in range(14):
        angle = 2 * math.pi * index / 14
        cx = seal_w / 2 + math.cos(angle) * (seal_w / 2 - 70)
        cy = seal_h / 2 + math.sin(angle) * (seal_h / 2 - 52)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=ink_line)
    rotated = seal.rotate(13, resample=Image.Resampling.BICUBIC, expand=True)
    x1, y1, x2, y2 = page.seal_center_bbox
    center_x = _to_px((x1 + x2) / 2)
    center_y = _to_px((y1 + y2) / 2)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste(
        rotated,
        (int(round(center_x - rotated.width / 2)), int(round(center_y - rotated.height / 2))),
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _draw_ink_smudges(
    image: Image.Image, boxes: list[tuple[float, float, float, float]], rng: random.Random
) -> Image.Image:
    """Cover printed fragments with dense, opaque stamp ink.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB page image with the seal already composited.
    boxes : list of tuple of float
        Pixel boxes ``(x1, y1, x2, y2)`` of the occluded text fragments.
    rng : random.Random
        Seeded generator so the blot pattern is deterministic.

    Returns
    -------
    PIL.Image.Image
        New RGB image where the fragments are hidden under overlapping ink
        blots (a near-opaque core with a softer halo), as when a rubber
        stamp is pressed with too much ink.
    """
    if not boxes:
        return image
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        blots = max(SMUDGE_BLOTS_PER_FRAGMENT, int(width / height * 9))
        for _ in range(blots):
            cx = x1 + rng.uniform(0.0, 1.0) * width
            cy = y1 + rng.uniform(0.3, 0.7) * height
            rx = rng.uniform(0.6, 1.3) * height
            ry = rng.uniform(0.55, 0.9) * height
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*SMUDGE_INK, SMUDGE_HALO_ALPHA // 4))
        for _ in range(blots * 2):
            cx = x1 + rng.uniform(0.02, 0.98) * width
            cy = y1 + rng.uniform(0.35, 0.65) * height
            rx = rng.uniform(0.25, 0.55) * height
            ry = rng.uniform(0.3, 0.5) * height
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*SMUDGE_INK, SMUDGE_CORE_ALPHA))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def render_raster_page(page: RasterPageSpec) -> Image.Image:
    """Render a scanned page image deterministically.

    Parameters
    ----------
    page : RasterPageSpec
        Raster page specification.

    Returns
    -------
    PIL.Image.Image
        1700 x 2200 RGB image (Letter at 200 dpi).

    Raises
    ------
    PipelineError
        If a region's text does not fit inside its bounding box, or an
        ``ink_smudge_fragments`` entry does not appear on any printed line.
    """
    rng = random.Random(page.noise_seed)
    image = Image.new("RGB", (RASTER_WIDTH_PX, RASTER_HEIGHT_PX), (246, 244, 238))
    draw = ImageDraw.Draw(image)
    font = _load_font(RASTER_FONT_PX)
    smudge_boxes: list[tuple[float, float, float, float]] = []
    for region in page.regions:
        x1, y1, x2, y2 = (_to_px(value) for value in region.bbox)
        lines = _wrap_words(region.text, x2 - x1, lambda value: draw.textlength(value, font=font))
        top = y1 + 4.0
        if top + len(lines) * RASTER_LINE_PX > y2:
            raise PipelineError(
                "raster region text does not fit its bounding box",
                {"text": region.text[:60], "lines": len(lines)},
            )
        for line in lines:
            draw.text((x1, top), line, font=font, fill=(28, 28, 34))
            for fragment in page.ink_smudge_fragments:
                index = line.find(fragment)
                if index >= 0:
                    start = x1 + draw.textlength(line[:index], font=font)
                    end = start + draw.textlength(fragment, font=font)
                    smudge_boxes.append((start, top, end, top + RASTER_LINE_PX))
            top += RASTER_LINE_PX
    missing = [
        fragment
        for fragment in page.ink_smudge_fragments
        if not any(fragment in region.text for region in page.regions)
    ]
    if missing or len(smudge_boxes) < len(page.ink_smudge_fragments):
        raise PipelineError(
            "ink smudge fragment not found on a single printed line",
            {"fragments": list(page.ink_smudge_fragments), "missing": missing},
        )
    for offset in (0, 1):
        points = [
            (1050 + step * 12, 1900 + offset + int(28 * math.sin(step / 3.0)) + rng.randint(-6, 6))
            for step in range(36)
        ]
        draw.line(points, fill=(40, 40, 60), width=3)
    draw.line((980, 1960, 1520, 1960), fill=(70, 70, 70), width=2)
    for _ in range(14):
        y = rng.randint(0, RASTER_HEIGHT_PX - 1)
        shade = rng.randint(222, 236)
        draw.line((0, y, RASTER_WIDTH_PX, y), fill=(shade, shade, shade - 4), width=1)
    for _ in range(9000):
        x = rng.randint(0, RASTER_WIDTH_PX - 1)
        y = rng.randint(0, RASTER_HEIGHT_PX - 1)
        tone = rng.randint(90, 200)
        draw.point((x, y), fill=(tone, tone, tone))
    return _draw_ink_smudges(_draw_seal(image, page), smudge_boxes, rng)


# Section: pdf assembly and io


def render_deed_pdf(spec: DeedSpec) -> bytes:
    """Render a deed specification into deterministic PDF bytes.

    Parameters
    ----------
    spec : DeedSpec
        Deed specification.

    Returns
    -------
    bytes
        PDF binary.
    """
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=LETTER, invariant=1, pageCompression=1)
    canvas.setTitle(spec.title)
    canvas.setAuthor("Synthetic Deed Generator")
    canvas.setSubject(spec.scenario)
    canvas.setCreator("actas-pipeline synthetic_deeds")
    for page in spec.pages:
        if isinstance(page, NativePageSpec):
            _draw_native_page(canvas, page)
        else:
            canvas.drawImage(
                ImageReader(render_raster_page(page)), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT
            )
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes atomically via a temporary file and ``os.replace``.

    Parameters
    ----------
    path : Path
        Destination file.
    data : bytes
        Content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def _write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON (``ensure_ascii=False``, ``indent=2``) atomically.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : Any
        JSON-serializable payload.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _write_bytes_atomic(path, text.encode("utf-8"))


def build_cache_payload(spec: DeedSpec, doc_sha256: str) -> dict[str, Any]:
    """Build the pre-verified cache payload of a deed.

    Parameters
    ----------
    spec : DeedSpec
        Deed specification.
    doc_sha256 : str
        SHA-256 of the rendered PDF.

    Returns
    -------
    dict of str to Any
        Payload in the contract cache format.
    """
    raw = RAW_EXTRACTION_BUILDERS[spec.acta_id]()
    validated = RawDeedExtraction.model_validate(raw.model_dump(by_alias=True, mode="json"))
    ocr_pages = {
        str(number): build_ocr_result(number, page).model_dump(mode="json")
        for number, page in enumerate(spec.pages, start=1)
        if isinstance(page, RasterPageSpec)
    }
    return {
        "doc_sha256": doc_sha256,
        "document_filename": spec.filename,
        "acta_id": spec.acta_id,
        "document_type": DOCUMENT_TYPE,
        "model_version": CACHE_MODEL_VERSION,
        "raw_extraction": validated.model_dump(by_alias=True, mode="json"),
        "ocr_pages": ocr_pages,
        "vlm_reads": build_vlm_reads(spec),
        "verifier_responses": build_verifier_responses(spec),
        "decision_scores": build_decision_scores(spec),
    }


def _remove_stale_cache_entries(settings: Settings, acta_id: str, keep_sha256: str) -> list[str]:
    """Delete pre-verified cache files of an acta left by older renders.

    Parameters
    ----------
    settings : Settings
        Runtime settings.
    acta_id : str
        Deed identifier.
    keep_sha256 : str
        Hash of the current render, which is kept.

    Returns
    -------
    list of str
        Hashes of the removed cache entries.
    """
    removed: list[str] = []
    if not settings.cache_dir.is_dir():
        return removed
    for candidate in sorted(settings.cache_dir.glob("*.json")):
        if candidate.stem == keep_sha256:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("acta_id") == acta_id
            and payload.get("model_version") == CACHE_MODEL_VERSION
        ):
            candidate.unlink()
            removed.append(candidate.stem)
    return removed


def _ground_truth_entry(spec: DeedSpec, doc_sha256: str) -> dict[str, Any]:
    """Build the ground-truth entry of a deed.

    Parameters
    ----------
    spec : DeedSpec
        Deed specification.
    doc_sha256 : str
        SHA-256 of the rendered PDF.

    Returns
    -------
    dict of str to Any
        Entry in the documented ground-truth format.
    """
    native_pages = [n for n, p in enumerate(spec.pages, start=1) if isinstance(p, NativePageSpec)]
    scanned_pages = [n for n, p in enumerate(spec.pages, start=1) if isinstance(p, RasterPageSpec)]
    entry: dict[str, Any] = {
        "acta_id": spec.acta_id,
        "document_filename": spec.filename,
        "doc_sha256": doc_sha256,
        "total_pages": spec.total_pages,
        "scenario": spec.scenario,
        "native_pages": native_pages,
        "scanned_pages": scanned_pages,
    }
    entry.update(build_ground_truth_fields(spec.acta_id))
    return entry


def generate_synthetic_deeds(settings: Settings) -> list[GeneratedDeed]:
    """Generate the three deeds, their cache payloads and the ground truth.

    Parameters
    ----------
    settings : Settings
        Runtime settings providing ``samples_dir``, ``cache_dir`` and
        ``ground_truth_path``.

    Returns
    -------
    list of GeneratedDeed
        One result per deed, in ``acta_a``, ``acta_b``, ``acta_c`` order.

    Raises
    ------
    PipelineError
        If rendering fails or the post-generation grounding and clause
        self-checks report any problem.

    Notes
    -----
    Stale pre-verified cache entries of the same acta (from a previous
    render with different bytes) are removed so the cache never holds
    orphaned payloads.
    """
    logger = get_logger("generators")
    results: list[GeneratedDeed] = []
    ground_truth: dict[str, Any] = {}
    for spec in DEED_SPECS:
        started = time.perf_counter()
        pdf_bytes = render_deed_pdf(spec)
        doc_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        pdf_path = settings.samples_dir / spec.filename
        _write_bytes_atomic(pdf_path, pdf_bytes)
        cache_path = settings.cache_file_for(doc_sha256)
        _write_json(cache_path, build_cache_payload(spec, doc_sha256))
        removed = _remove_stale_cache_entries(settings, spec.acta_id, doc_sha256)
        ground_truth[spec.acta_id] = _ground_truth_entry(spec, doc_sha256)
        results.append(
            GeneratedDeed(
                acta_id=spec.acta_id,
                pdf_path=pdf_path,
                sha256=doc_sha256,
                cache_path=cache_path,
                total_pages=spec.total_pages,
            )
        )
        log_event(
            logger,
            tag=StatusTag.OK,
            event="synthetic_deed_written",
            stage=_STAGE,
            doc_sha256=doc_sha256,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            details={
                "acta_id": spec.acta_id,
                "pdf_path": str(pdf_path),
                "cache_path": str(cache_path),
                "total_pages": spec.total_pages,
                "bytes": len(pdf_bytes),
                "stale_cache_removed": removed,
            },
            message=f"{spec.filename} written ({spec.total_pages} pages)",
        )
    _write_json(settings.ground_truth_path, ground_truth)
    log_event(
        logger,
        tag=StatusTag.OK,
        event="ground_truth_written",
        stage=_STAGE,
        details={"path": str(settings.ground_truth_path), "actas": sorted(ground_truth)},
        message=f"ground truth written for {len(ground_truth)} deeds",
    )
    problems = verify_cache_grounding(settings) + verify_mandated_clauses(settings)
    if problems:
        raise PipelineError("synthetic deed self-check failed", {"problems": problems})
    log_event(
        logger,
        tag=StatusTag.OK,
        event="synthetic_selfcheck_passed",
        stage=_STAGE,
        details={"deeds": len(results)},
        message="grounding and mandated clause self-checks passed",
    )
    return results


# Section: self-checks


def iter_raw_evidence(raw: RawDeedExtraction) -> Iterator[tuple[str, Evidence]]:
    """Yield every evidence object in a raw extraction with a label.

    Parameters
    ----------
    raw : RawDeedExtraction
        Extraction to traverse.

    Yields
    ------
    tuple of (str, Evidence)
        Label (field path) and evidence, covering the 8 fields, their list
        items and the amendment events.
    """
    fields: dict[str, FieldWithEvidence[Any]] = {
        "razon_social": raw.razon_social,
        "fecha_constitucion": raw.fecha_constitucion,
        "numero_escritura": raw.numero_escritura,
        "notario": raw.notario,
        "socios": raw.socios,
        "representantes_legales": raw.representantes_legales,
        "poderes": raw.poderes,
        "capital_social_actual": raw.capital_social_actual,
    }
    for name, field_value in fields.items():
        if field_value.evidence is not None:
            yield name, field_value.evidence
    for index, socio in enumerate(raw.socios.value or []):
        yield f"socios[{index}]", socio.evidence
    for index, rep in enumerate(raw.representantes_legales.value or []):
        yield f"representantes_legales[{index}]", rep.evidence
    for index, power in enumerate(raw.poderes.value or []):
        yield f"poderes[{index}]", power.evidence
    for index, event in enumerate(raw.amendment_history):
        yield f"amendment_history[{index}]", event.evidence


def _page_texts(pdf_path: Path) -> list[str]:
    """Extract native text of every page with PyMuPDF.

    Parameters
    ----------
    pdf_path : Path
        PDF file.

    Returns
    -------
    list of str
        Native text per page (index 0 is page 1).
    """
    document: Any = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    try:
        return [str(document.load_page(index).get_text()) for index in range(document.page_count)]
    finally:
        document.close()


def verify_cache_grounding(settings: Settings) -> list[str]:
    """Check that every cached source quote is grounded in its page.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    list of str
        Human-readable problems; empty when all checks pass.

    Notes
    -----
    For each deed: the PDF exists, its SHA-256 has a cache file with
    matching metadata, the page count matches, native pages have at least
    ``native_char_threshold`` alphanumeric characters, scanned pages have
    zero native alphanumeric characters and a cached OCR text, and every
    ``source_quote`` is a substring of the whitespace-normalized text of its
    page (OCR text for scanned pages).
    """
    problems: list[str] = []
    for spec in DEED_SPECS:
        pdf_path = settings.samples_dir / spec.filename
        if not pdf_path.is_file():
            problems.append(f"{spec.acta_id}: missing PDF {pdf_path}")
            continue
        doc_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        cache_path = settings.cache_file_for(doc_sha256)
        if not cache_path.is_file():
            problems.append(f"{spec.acta_id}: missing cache file {cache_path.name}")
            continue
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("doc_sha256", doc_sha256),
            ("acta_id", spec.acta_id),
            ("document_filename", spec.filename),
            ("model_version", CACHE_MODEL_VERSION),
        ):
            if payload.get(key) != expected:
                problems.append(f"{spec.acta_id}: cache {key} mismatch")
        native_texts = _page_texts(pdf_path)
        if len(native_texts) != spec.total_pages:
            problems.append(f"{spec.acta_id}: expected {spec.total_pages} pages, got {len(native_texts)}")
            continue
        page_texts: dict[int, str] = {}
        for number, (page_spec, native_text) in enumerate(zip(spec.pages, native_texts), start=1):
            clean = count_clean_chars(native_text)
            if isinstance(page_spec, RasterPageSpec):
                if clean != 0:
                    problems.append(f"{spec.acta_id} p{number}: scanned page has {clean} native chars")
                ocr_entry = payload.get("ocr_pages", {}).get(str(number))
                if not isinstance(ocr_entry, dict) or not ocr_entry.get("text"):
                    problems.append(f"{spec.acta_id} p{number}: missing cached OCR text")
                    page_texts[number] = ""
                else:
                    page_texts[number] = normalize_whitespace(str(ocr_entry["text"]))
            else:
                if clean < settings.native_char_threshold:
                    problems.append(f"{spec.acta_id} p{number}: only {clean} native chars")
                page_texts[number] = normalize_whitespace(native_text)
        raw = RawDeedExtraction.model_validate(payload["raw_extraction"])
        for label, evidence in iter_raw_evidence(raw):
            text = page_texts.get(evidence.page_number)
            if text is None:
                problems.append(f"{spec.acta_id} {label}: page {evidence.page_number} out of range")
            elif normalize_whitespace(evidence.source_quote) not in text:
                problems.append(
                    f"{spec.acta_id} {label}: quote not found on page {evidence.page_number}: "
                    f"{evidence.source_quote[:70]!r}"
                )
    return problems


def verify_mandated_clauses(settings: Settings) -> list[str]:
    """Check that every spec-mandated clause appears verbatim on its page.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    list of str
        Human-readable problems; empty when all clauses are present.

    Notes
    -----
    Native clauses are checked against PyMuPDF text (whitespace
    normalized); raster clauses against the cached OCR text, since the
    scanned page has no native text by design.
    """
    problems: list[str] = []
    for spec in DEED_SPECS:
        pdf_path = settings.samples_dir / spec.filename
        if not pdf_path.is_file():
            problems.append(f"{spec.acta_id}: missing PDF {pdf_path}")
            continue
        payload_path = settings.cache_file_for(hashlib.sha256(pdf_path.read_bytes()).hexdigest())
        ocr_pages: dict[str, Any] = {}
        if payload_path.is_file():
            ocr_pages = json.loads(payload_path.read_text(encoding="utf-8")).get("ocr_pages", {})
        native_texts = _page_texts(pdf_path)
        for number, page_spec in enumerate(spec.pages, start=1):
            if isinstance(page_spec, NativePageSpec):
                text = normalize_whitespace(native_texts[number - 1]) if number <= len(native_texts) else ""
                clauses = [block.text for block in page_spec.blocks if block.mandated]
            else:
                text = normalize_whitespace(str(ocr_pages.get(str(number), {}).get("text", "")))
                clauses = [region.text for region in page_spec.regions if region.mandated]
            for clause in clauses:
                if normalize_whitespace(clause) not in text:
                    problems.append(f"{spec.acta_id} p{number}: mandated clause missing: {clause[:70]!r}")
    return problems
