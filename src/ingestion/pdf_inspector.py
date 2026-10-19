"""Page-level PDF inspection and routing.

Each page is inspected independently with PyMuPDF. Pages with enough clean
alphanumeric native characters are extracted natively; the rest are rendered
to PNG at 200 dpi for OCR.
"""

from __future__ import annotations

import time

import pymupdf

from src.core.config import Settings
from src.core.exceptions import PDFInspectionError
from src.core.logger import StatusTag, get_logger, log_event
from src.providers.interfaces import PageContent, PageSource
from src.schemas.deed_models import BoundingBox, ConfidenceSource

RENDER_DPI: int = 200
"""Resolution used when rendering scanned pages for OCR."""

_logger = get_logger("ingestion.pdf_inspector")


def count_clean_characters(text: str) -> int:
    """Count alphanumeric characters in a text.

    Parameters
    ----------
    text : str
        Raw native page text.

    Returns
    -------
    int
        Number of characters for which ``str.isalnum`` is true.
    """
    return sum(1 for char in text if char.isalnum())


class PDFInspector:
    """Route each PDF page to native extraction or the OCR path.

    Parameters
    ----------
    settings : Settings
        Pipeline settings; ``native_char_threshold`` drives routing.

    Notes
    -----
    Native pages carry confidence ``1.0`` with ``ConfidenceSource.NATIVE_PDF``.
    Scanned pages carry empty text, confidence ``0.0``, source
    ``OCR_ENGINE`` and the rendered PNG until OCR fills them in.
    """

    def __init__(self, settings: Settings) -> None:
        """Store the settings.

        Parameters
        ----------
        settings : Settings
            Pipeline settings.
        """
        self._settings = settings

    def inspect(
        self, pdf_bytes: bytes, *, job_id: str = "-", doc_sha256: str = "-"
    ) -> list[PageContent]:
        """Inspect every page of a PDF.

        Parameters
        ----------
        pdf_bytes : bytes
            PDF binary.
        job_id : str, default "-"
            Job identifier for audit records.
        doc_sha256 : str, default "-"
            Document hash for audit records.

        Returns
        -------
        list of PageContent
            One entry per page in page order (1-indexed).

        Raises
        ------
        PDFInspectionError
            If the bytes are not a readable PDF, the PDF has no pages, or
            a page cannot be read or rendered.
        """
        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise PDFInspectionError(
                "payload is not a readable PDF", {"error": str(exc)}
            ) from exc
        pages: list[PageContent] = []
        try:
            if document.page_count == 0:
                raise PDFInspectionError("PDF contains no pages")
            for index in range(document.page_count):
                pages.append(self._inspect_page(document, index, job_id, doc_sha256))
        finally:
            document.close()  # type: ignore[no-untyped-call]
        return pages

    def _inspect_page(
        self, document: pymupdf.Document, index: int, job_id: str, doc_sha256: str
    ) -> PageContent:
        """Inspect a single page and log the routing decision.

        Parameters
        ----------
        document : pymupdf.Document
            Open document.
        index : int
            Zero-based page index.
        job_id : str
            Job identifier for audit records.
        doc_sha256 : str
            Document hash for audit records.

        Returns
        -------
        PageContent
            Routed page content.

        Raises
        ------
        PDFInspectionError
            If text extraction or rendering fails.
        """
        started = time.perf_counter()
        page_number = index + 1
        try:
            page = document.load_page(index)  # type: ignore[no-untyped-call]
            text = str(page.get_text("text"))
            clean = count_clean_characters(text)
            if clean >= self._settings.native_char_threshold:
                content = PageContent(
                    page_number=page_number,
                    source=PageSource.NATIVE_PDF,
                    text=text,
                    clean_char_count=clean,
                    confidence_score=1.0,
                    confidence_source=ConfidenceSource.NATIVE_PDF,
                )
            else:
                pixmap = page.get_pixmap(dpi=RENDER_DPI)
                content = PageContent(
                    page_number=page_number,
                    source=PageSource.SCANNED_PAGE,
                    text="",
                    clean_char_count=clean,
                    confidence_score=0.0,
                    confidence_source=ConfidenceSource.OCR_ENGINE,
                    image_png=pixmap.tobytes("png"),
                )
        except PDFInspectionError:
            raise
        except Exception as exc:
            raise PDFInspectionError(
                f"failed to inspect page {page_number}",
                {"page_number": page_number, "error": str(exc)},
            ) from exc
        duration_ms = (time.perf_counter() - started) * 1000.0
        routed_native = content.source is PageSource.NATIVE_PDF
        log_event(
            _logger,
            tag=StatusTag.INFO,
            event="page_routed",
            stage="inspection",
            job_id=job_id,
            doc_sha256=doc_sha256,
            duration_ms=duration_ms,
            details={
                "page_number": page_number,
                "clean_char_count": clean,
                "threshold": self._settings.native_char_threshold,
                "route": content.source.value,
                "image_rendered": content.image_png is not None,
            },
            message=(
                f"page {page_number} routed to "
                f"{'native extraction' if routed_native else 'OCR'} "
                f"({clean} clean characters)"
            ),
        )
        return content


def render_region_png(
    pdf_bytes: bytes,
    page_number: int,
    bbox: BoundingBox,
    *,
    dpi: int = RENDER_DPI,
    padding: float = 12.0,
) -> bytes:
    """Render a padded page region as a PNG crop.

    Parameters
    ----------
    pdf_bytes : bytes
        PDF binary.
    page_number : int
        1-indexed page number.
    bbox : BoundingBox
        Region in PDF points (top-left origin, as used by PyMuPDF).
    dpi : int, default 200
        Render resolution.
    padding : float, default 12.0
        Margin in points added on every side before clamping to the page.

    Returns
    -------
    bytes
        PNG image of the clipped region.

    Raises
    ------
    PDFInspectionError
        If the PDF cannot be opened, the page does not exist, or the padded
        region does not intersect the page.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise PDFInspectionError("payload is not a readable PDF", {"error": str(exc)}) from exc
    try:
        if page_number < 1 or page_number > document.page_count:
            raise PDFInspectionError(
                f"page {page_number} does not exist",
                {"page_number": page_number, "page_count": document.page_count},
            )
        page = document.load_page(page_number - 1)  # type: ignore[no-untyped-call]
        bounds = page.rect
        clip = pymupdf.Rect(  # type: ignore[no-untyped-call]
            max(bounds.x0, bbox.x1 - padding),
            max(bounds.y0, bbox.y1 - padding),
            min(bounds.x1, bbox.x2 + padding),
            min(bounds.y1, bbox.y2 + padding),
        )
        if clip.is_empty:
            raise PDFInspectionError(
                "crop region does not intersect the page",
                {"page_number": page_number, "bbox": bbox.as_list()},
            )
        return bytes(page.get_pixmap(dpi=dpi, clip=clip).tobytes("png"))
    except PDFInspectionError:
        raise
    except Exception as exc:
        raise PDFInspectionError(
            f"failed to render region on page {page_number}",
            {"page_number": page_number, "error": str(exc)},
        ) from exc
    finally:
        document.close()  # type: ignore[no-untyped-call]


def page_bbox(pdf_bytes: bytes, page_number: int) -> BoundingBox:
    """Return the full-page bounding box of a PDF page.

    Parameters
    ----------
    pdf_bytes : bytes
        PDF binary.
    page_number : int
        1-indexed page number.

    Returns
    -------
    BoundingBox
        Page rectangle in PDF points (top-left origin).

    Raises
    ------
    PDFInspectionError
        If the PDF cannot be opened or the page does not exist.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise PDFInspectionError("payload is not a readable PDF", {"error": str(exc)}) from exc
    try:
        if page_number < 1 or page_number > document.page_count:
            raise PDFInspectionError(
                f"page {page_number} does not exist",
                {"page_number": page_number, "page_count": document.page_count},
            )
        bounds = document.load_page(page_number - 1).rect  # type: ignore[no-untyped-call]
        return BoundingBox(
            x1=float(bounds.x0), y1=float(bounds.y0), x2=float(bounds.x1), y2=float(bounds.y1)
        )
    finally:
        document.close()  # type: ignore[no-untyped-call]
