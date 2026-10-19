"""Lean provider abstractions and page-level transfer models.

High-level pipeline components depend only on these interfaces
(Dependency Inversion); concrete OCR, verifier, LLM and storage
implementations live in sibling modules and may be live or
deterministic (cache-backed). The vision reader re-reads OCR regions from
their image crop (low-confidence regions before extraction, critical-field
regions after it). The decision scorer answers ordinal ``score`` questions
and is used only by the ``test-decisions`` capability demo, never by the
processing path; it is kept separate from the predicate verifier
(Interface Segregation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

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


class ScoreLevel(BaseModel):
    """One ordered level of a Decisions ``score`` question.

    Parameters
    ----------
    label : str
        Short level label.
    description : str
        What the level means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    description: str


class ScoreQuestion(BaseModel):
    """A Decisions ``score`` question with levels ordered lowest to highest.

    Parameters
    ----------
    name : str
        Question name echoed in the answer (for example ``PROMPT_INJECTION_RISK_SCORE``).
    instructions : str
        Rating instructions.
    levels : tuple of ScoreLevel
        At least two levels, lowest first; level ``i`` has index ``i``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    instructions: str
    levels: tuple[ScoreLevel, ...] = Field(min_length=2)


class ScoreLevelProbability(BaseModel):
    """Probability assigned to one level of a ``score`` question.

    Parameters
    ----------
    value : int
        0-based level index.
    label : str
        Level label.
    probability : float
        Probability of the level.
    """

    model_config = ConfigDict(extra="forbid")

    value: int = Field(ge=0)
    label: str
    probability: float = Field(ge=0.0, le=1.0)


ScoreOutcome = Literal["answered", "refused", "unknown"]
"""Outcome of a score question: answered, refused by the model, or unknown (no cached answer)."""


class ScoreResult(BaseModel):
    """Answer to one Decisions ``score`` question about a page.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    name : str
        Question name.
    outcome : {"answered", "refused", "unknown"}
        ``refused`` when the model declined; ``unknown`` when a
        deterministic provider has no cached answer.
    score : float or None, default None
        Probability-weighted average of the 0-based level indices.
    normalized_score : float or None, default None
        ``score / (levels - 1)`` in ``[0.0, 1.0]``.
    confidence : float or None, default None
        Confidence reported by the API.
    probabilities : list of ScoreLevelProbability
        Per-level probabilities.
    levels : int
        Number of levels of the question.
    engine : str
        Scorer identifier.
    input_tokens : int or None, default None
        Prompt tokens of the request that produced the answer (shared by
        every question batched in that request).
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    name: str
    outcome: ScoreOutcome
    score: Optional[float] = Field(default=None, ge=0.0)
    normalized_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    probabilities: list[ScoreLevelProbability] = Field(default_factory=list)
    levels: int = Field(ge=2)
    engine: str
    input_tokens: Optional[int] = None


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


class BaseDecisionScorer(ABC):
    """Abstract scorer answering Decisions ``score`` questions (``test-decisions`` only)."""

    @abstractmethod
    def score_page(
        self,
        doc_sha256: str,
        page_number: int,
        questions: list[ScoreQuestion],
        *,
        text: str,
        image_png: Optional[bytes] = None,
    ) -> list[ScoreResult]:
        """Answer independent score questions about one page in one request.

        Parameters
        ----------
        doc_sha256 : str
            Document hash (audit records).
        page_number : int
            1-indexed page number.
        questions : list of ScoreQuestion
            Questions answered over the same input, batched together.
        text : str
            Untrusted text.
        image_png : bytes or None, default None
            Optional image attached to the same request.

        Returns
        -------
        list of ScoreResult
            One result per question, in question order; a refusal is
            returned as ``outcome="refused"`` and never raised.

        Raises
        ------
        src.core.exceptions.DecisionScoreError
            If the request fails or the response is malformed.
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
