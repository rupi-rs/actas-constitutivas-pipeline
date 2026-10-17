"""OCR providers for scanned pages.

``DeterministicOCRProvider`` serves pre-verified OCR output from the
SHA-256 cache; ``RapidOCRProvider`` runs a local engine when the optional
``rapidocr-onnxruntime`` package is installed.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.exceptions import OCRError, StorageError
from src.core.logger import StatusTag, get_logger, log_event
from src.providers.interfaces import (
    BaseOCRProvider,
    BaseStorageRepository,
    OCRRegion,
    OCRResult,
)
from src.schemas.deed_models import BoundingBox

_logger = get_logger("providers.ocr")

_PIXELS_TO_POINTS: float = 72.0 / 200.0
"""Conversion from 200 dpi pixels to PDF points."""


class DeterministicOCRProvider(BaseOCRProvider):
    """Cache-backed OCR provider.

    Parameters
    ----------
    repository : BaseStorageRepository
        Repository holding ``cache["ocr_pages"][str(page_number)]``.
    """

    def __init__(self, repository: BaseStorageRepository) -> None:
        """Store the repository.

        Parameters
        ----------
        repository : BaseStorageRepository
            Cache repository.
        """
        self._repository = repository

    def extract_page(self, doc_sha256: str, page_number: int, image_png: bytes) -> OCRResult:
        """Return the cached OCR result for a page.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            1-indexed page number.
        image_png : bytes
            Ignored; accepted for interface compatibility.

        Returns
        -------
        OCRResult
            Cached OCR output.

        Raises
        ------
        OCRError
            If the cache entry or the page entry is missing or invalid.
        """
        started = time.perf_counter()
        try:
            cache = self._repository.load_cache(doc_sha256)
        except StorageError as exc:
            raise OCRError(f"cannot read OCR cache: {exc.message}", exc.details) from exc
        if cache is None:
            raise OCRError(
                "no pre-verified cache entry for this document; deterministic OCR is unavailable",
                {"doc_sha256": doc_sha256, "page_number": page_number},
            )
        entry = (cache.get("ocr_pages") or {}).get(str(page_number))
        if entry is None:
            raise OCRError(
                f"no cached OCR result for page {page_number}",
                {"doc_sha256": doc_sha256, "page_number": page_number},
            )
        try:
            result = OCRResult.model_validate(entry)
        except ValueError as exc:
            raise OCRError(
                f"cached OCR result for page {page_number} is invalid",
                {"doc_sha256": doc_sha256, "page_number": page_number, "error": str(exc)},
            ) from exc
        log_event(
            _logger,
            tag=StatusTag.INFO,
            event="ocr_cache_hit",
            stage="ocr",
            doc_sha256=doc_sha256,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={
                "page_number": page_number,
                "engine": result.engine,
                "mean_confidence": result.mean_confidence,
                "regions": len(result.regions),
            },
            message=f"page {page_number} OCR served from cache (mean confidence {result.mean_confidence:.2f})",
        )
        return result


class RapidOCRProvider(BaseOCRProvider):
    """Local OCR provider backed by ``rapidocr-onnxruntime``.

    Raises
    ------
    OCRError
        On construction if the optional package is not installed.

    Notes
    -----
    The engine is imported lazily. Pixel boxes at 200 dpi are converted to
    PDF points so they are comparable with native coordinates.
    """

    ENGINE_NAME: str = "rapidocr-onnxruntime"

    def __init__(self) -> None:
        """Import and initialise the engine.

        Raises
        ------
        OCRError
            If ``rapidocr_onnxruntime`` cannot be imported or initialised.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OCRError(
                "rapidocr-onnxruntime is not installed; scanned pages cannot be recognised "
                "live. Run 'uv sync --extra ocr' or process the document from its cache",
                {"package": "rapidocr-onnxruntime"},
            ) from exc
        try:
            self._engine: Any = RapidOCR()
        except Exception as exc:
            raise OCRError("failed to initialise RapidOCR", {"error": str(exc)}) from exc

    def extract_page(self, doc_sha256: str, page_number: int, image_png: bytes) -> OCRResult:
        """Run OCR on a rendered page.

        Parameters
        ----------
        doc_sha256 : str
            Document hash (for audit records).
        page_number : int
            1-indexed page number.
        image_png : bytes
            PNG rendering at 200 dpi.

        Returns
        -------
        OCRResult
            Recognised text and regions with boxes in PDF points.

        Raises
        ------
        OCRError
            If the engine fails or the image is empty.
        """
        if not image_png:
            raise OCRError("empty page image", {"page_number": page_number})
        started = time.perf_counter()
        try:
            raw, _elapsed = self._engine(image_png)
        except Exception as exc:
            raise OCRError(
                f"RapidOCR failed on page {page_number}",
                {"page_number": page_number, "error": str(exc)},
            ) from exc
        regions: list[OCRRegion] = []
        for item in raw or []:
            points, text, score = item[0], str(item[1]), float(item[2])
            xs = [float(point[0]) * _PIXELS_TO_POINTS for point in points]
            ys = [float(point[1]) * _PIXELS_TO_POINTS for point in points]
            regions.append(
                OCRRegion(
                    text=text,
                    bbox=BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
                    confidence=min(1.0, max(0.0, score)),
                )
            )
        mean = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        result = OCRResult(
            page_number=page_number,
            text="\n".join(r.text for r in regions),
            mean_confidence=mean,
            regions=regions,
            engine=self.ENGINE_NAME,
        )
        log_event(
            _logger,
            tag=StatusTag.INFO,
            event="ocr_completed",
            stage="ocr",
            doc_sha256=doc_sha256,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={
                "page_number": page_number,
                "engine": self.ENGINE_NAME,
                "mean_confidence": mean,
                "regions": len(regions),
            },
            message=f"page {page_number} recognised by {self.ENGINE_NAME} ({len(regions)} regions)",
        )
        return result


def build_ocr_provider(
    repository: BaseStorageRepository, engine: str = "deterministic"
) -> BaseOCRProvider:
    """Create an OCR provider by engine name.

    Parameters
    ----------
    repository : BaseStorageRepository
        Cache repository for the deterministic provider.
    engine : str, default "deterministic"
        ``"deterministic"`` or ``"rapidocr"``.

    Returns
    -------
    BaseOCRProvider
        The selected provider.

    Raises
    ------
    OCRError
        If the engine name is unknown or the engine is unavailable.
    """
    key = engine.strip().lower()
    if key == "deterministic":
        return DeterministicOCRProvider(repository)
    if key == "rapidocr":
        return RapidOCRProvider()
    raise OCRError(f"unknown OCR engine: {engine}", {"engine": engine})
