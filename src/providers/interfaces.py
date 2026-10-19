"""Lean provider abstractions and page-level transfer models.

High-level pipeline components depend only on these interfaces
(Dependency Inversion); concrete OCR, verifier, LLM and storage
implementations live in sibling modules and may be live or
deterministic (cache-backed). The vision reader re-reads low-confidence OCR
regions from their image crop before structured extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.deed_models import (
    BoundingBox,
    ConfidenceSource,
    CorporateDeedOutput,
    RawDeedExtraction,
)


class PageSource(str, Enum):
    """Routing decision for a PDF page."""

    NATIVE_PDF = "native_pdf"
    SCANNED_PAGE = "scanned_page"


class PageContent(BaseModel):
    """Text content of one page after native extraction or OCR.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    source : PageSource
        Native or scanned routing decision.
    text : str
        Page text (native or OCR).
    clean_char_count : int
        Count of clean alphanumeric native characters.
    confidence_score : float
        Page-level confidence (``1.0`` for native pages).
    confidence_source : ConfidenceSource
        Origin of the confidence.
    image_png : bytes or None, default None
        Rendered page image for scanned pages; excluded from dumps.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    source: PageSource
    text: str
    clean_char_count: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_source: ConfidenceSource
    image_png: Optional[bytes] = Field(default=None, exclude=True, repr=False)


class OCRRegion(BaseModel):
    """A recognised text region.

    Parameters
    ----------
    text : str
        Region text.
    bbox : BoundingBox
        Region location in PDF points.
    confidence : float
        Engine confidence in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


class OCRResult(BaseModel):
    """OCR output for one page.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    text : str
        Full page text.
    mean_confidence : float
        Mean region confidence.
    regions : list of OCRRegion
        Recognised regions.
    engine : str
        Engine identifier (for example ``"deterministic-cache"``).
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    mean_confidence: float = Field(ge=0.0, le=1.0)
    regions: list[OCRRegion] = Field(default_factory=list)
    engine: str


class VerifierResult(BaseModel):
    """Escalation Tier 2 predicate verification result.

    Parameters
    ----------
    predicate : str
        Natural-language predicate evaluated.
    probability : float
        Probability ``p`` that the predicate is true.
    calibrated_confidence : float
        ``|2p - 1|``.
    engine : str
        Verifier identifier.
    """

    model_config = ConfigDict(extra="forbid")

    predicate: str
    probability: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    engine: str


class VisionReadResult(BaseModel):
    """Vision-model re-read of one low-confidence OCR region.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    bbox : BoundingBox
        Region location in PDF points (the crop adds 12 pt padding).
    reading : str or None
        Text the model judged visibly legible; ``None`` when it declined.
    legible : bool
        ``False`` when any character needed to read the region is occluded.
    engine : str
        Reader identifier (for example ``"openai-vision:gpt-6-luna"``).
    input_tokens : int or None, default None
        Prompt tokens reported by a live reader.
    output_tokens : int or None, default None
        Completion tokens reported by a live reader.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    bbox: BoundingBox
    reading: Optional[str] = None
    legible: bool
    engine: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class BaseOCRProvider(ABC):
    """Abstract OCR provider for scanned pages."""

    @abstractmethod
    def extract_page(self, doc_sha256: str, page_number: int, image_png: bytes) -> OCRResult:
        """Recognise text on a rendered page image.

        Parameters
        ----------
        doc_sha256 : str
            Document hash (used by cache-backed providers).
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
        src.core.exceptions.OCRError
            If recognition fails.
        """


class BaseVerifierProvider(ABC):
    """Abstract predicate verifier (Escalation Tier 2)."""

    @abstractmethod
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
        """Estimate the probability that a predicate holds.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            1-indexed page number.
        predicate : str
            Statement to verify.
        context_text : str
            Page or region text supporting the decision.
        bbox : BoundingBox or None
            Region of interest.
        image_png : bytes or None, default None
            PNG crop of the region of interest for visual verification;
            providers that do not use images ignore it.

        Returns
        -------
        VerifierResult
            Probability and calibrated confidence ``|2p - 1|``.

        Raises
        ------
        src.core.exceptions.VerifierError
            If verification fails.
        """


class BaseVisionReader(ABC):
    """Abstract vision reader that re-reads a low-confidence region crop."""

    @abstractmethod
    def reread_region(
        self,
        doc_sha256: str,
        page_number: int,
        bbox: BoundingBox,
        image_png: Optional[bytes],
        ocr_text: str,
    ) -> VisionReadResult:
        """Transcribe the visibly legible text of a region crop.

        Parameters
        ----------
        doc_sha256 : str
            Document hash (used by cache-backed readers).
        page_number : int
            1-indexed page number.
        bbox : BoundingBox
            Region location in PDF points.
        image_png : bytes or None
            PNG crop of the padded region; live readers require it.
        ocr_text : str
            Untrusted Tier 1 OCR reading of the region (context only).

        Returns
        -------
        VisionReadResult
            Reading and legibility verdict.

        Raises
        ------
        src.core.exceptions.VisionReadError
            If the read fails, is refused or returns invalid data.
        """


class BaseLLMProvider(ABC):
    """Abstract structured-extraction LLM provider."""

    @abstractmethod
    def extract(
        self, doc_sha256: str, pages: list[PageContent], document_type: str
    ) -> tuple[RawDeedExtraction, str]:
        """Extract the pre-resolution structured deed.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        pages : list of PageContent
            Page texts to wrap in ``<page page_number="N">`` boundaries.
        document_type : str
            Registered document type identifier.

        Returns
        -------
        tuple of (RawDeedExtraction, str)
            Raw extraction and the model version string.

        Raises
        ------
        src.core.exceptions.ExtractionError
            If extraction fails or returns invalid data.
        """


class BaseStorageRepository(ABC):
    """Abstract persistence for the SHA-256 cache and final outputs."""

    @abstractmethod
    def load_cache(self, doc_sha256: str) -> Optional[dict[str, Any]]:
        """Load a cache entry.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.

        Returns
        -------
        dict of str to Any or None
            Cache payload, or ``None`` on a miss.
        """

    @abstractmethod
    def save_cache(self, doc_sha256: str, payload: dict[str, Any]) -> Path:
        """Persist a cache entry.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        payload : dict of str to Any
            Cache payload.

        Returns
        -------
        Path
            Written file path.

        Raises
        ------
        src.core.exceptions.StorageError
            If writing fails.
        """

    @abstractmethod
    def save_output(self, acta_id: str, output: CorporateDeedOutput) -> Path:
        """Persist a final output.

        Parameters
        ----------
        acta_id : str
            Output identifier (for example ``"acta_a"``).
        output : CorporateDeedOutput
            Final output.

        Returns
        -------
        Path
            Written file path.

        Raises
        ------
        src.core.exceptions.StorageError
            If writing fails.
        """

    @abstractmethod
    def save_crop(self, crop_name: str, image_png: bytes) -> str:
        """Persist a region crop sent to the vision reader and verifier.

        Parameters
        ----------
        crop_name : str
            Safe file name of the crop.
        image_png : bytes
            PNG crop.

        Returns
        -------
        str
            Location of the crop relative to the project root.

        Raises
        ------
        src.core.exceptions.StorageError
            If writing fails.
        """

    @abstractmethod
    def load_output(self, acta_id: str) -> Optional[dict[str, Any]]:
        """Load a final output as a JSON dictionary.

        Parameters
        ----------
        acta_id : str
            Output identifier.

        Returns
        -------
        dict of str to Any or None
            Output payload, or ``None`` if absent.
        """
