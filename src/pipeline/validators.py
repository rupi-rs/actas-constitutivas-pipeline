"""Deterministic post-extraction validators.

Implements the page-level grounding check, the numeric-versus-words capital
consistency check and confidence threshold routing (spec 03 section 2).
Every validator mutates the evidence / review flags of the model it
receives and works on both :class:`~src.schemas.deed_models.RawDeedExtraction`
and :class:`~src.schemas.deed_models.CorporateDeedOutput`.

Notes
-----
These checks are deliberately LLM-free: they make the extraction
auditable by proving each quote exists on the cited page and that the
capital amount is internally consistent.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from typing import Any, Optional, Union

from rapidfuzz import fuzz

from src.core.logger import StatusTag, get_logger, log_event
from src.schemas.deed_models import (
    CorporateDeedOutput,
    Evidence,
    FieldStatus,
    FieldWithEvidence,
    RawDeedExtraction,
)

DeedModel = Union[RawDeedExtraction, CorporateDeedOutput]
"""Models carrying the 8 contract fields accepted by every validator."""

_LOGGER = get_logger("pipeline.validators")
_STAGE: str = "validation"

GROUNDING_FAILURE_REASON: str = "Grounding check failed: quote not found on specified page"
"""Review reason appended when a quote is not found on its cited page."""

MISSING_EVIDENCE_REASON: str = "Grounding check failed: extracted value has no evidence"
"""Review reason appended when an extracted field carries no evidence."""

CAPITAL_MISMATCH_REASON: str = "Capital amount mismatch: numeric value does not match amount in words"
"""Review reason appended when the capital number and words disagree."""

GROUNDING_FAILED_CONFIDENCE_CAP: float = 0.40
"""Maximum confidence kept by evidence that fails the grounding check."""

SCALAR_FIELDS: tuple[str, ...] = (
    "razon_social",
    "fecha_constitucion",
    "numero_escritura",
    "notario",
    "capital_social_actual",
)
"""Contract fields whose value is a single object."""

LIST_FIELDS: tuple[str, ...] = ("socios", "representantes_legales", "poderes")
"""Contract fields whose value is a list of items with their own evidence."""

CONTRACT_FIELDS: tuple[str, ...] = (
    "razon_social",
    "fecha_constitucion",
    "numero_escritura",
    "notario",
    "socios",
    "representantes_legales",
    "poderes",
    "capital_social_actual",
)
"""The 8 contract fields in spec order."""

_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")

_UNITS: tuple[str, ...] = (
    "",
    "UN",
    "DOS",
    "TRES",
    "CUATRO",
    "CINCO",
    "SEIS",
    "SIETE",
    "OCHO",
    "NUEVE",
    "DIEZ",
    "ONCE",
    "DOCE",
    "TRECE",
    "CATORCE",
    "QUINCE",
    "DIECISEIS",
    "DIECISIETE",
    "DIECIOCHO",
    "DIECINUEVE",
    "VEINTE",
    "VEINTIUN",
    "VEINTIDOS",
    "VEINTITRES",
    "VEINTICUATRO",
    "VEINTICINCO",
    "VEINTISEIS",
    "VEINTISIETE",
    "VEINTIOCHO",
    "VEINTINUEVE",
)
_TENS: tuple[str, ...] = (
    "",
    "",
    "",
    "TREINTA",
    "CUARENTA",
    "CINCUENTA",
    "SESENTA",
    "SETENTA",
    "OCHENTA",
    "NOVENTA",
)
_HUNDREDS: tuple[str, ...] = (
    "",
    "CIENTO",
    "DOSCIENTOS",
    "TRESCIENTOS",
    "CUATROCIENTOS",
    "QUINIENTOS",
    "SEISCIENTOS",
    "SETECIENTOS",
    "OCHOCIENTOS",
    "NOVECIENTOS",
)
_MAX_AMOUNT: int = 999_999_999


def normalize_for_grounding(text: str) -> str:
    """Normalize text for quote-to-page comparison.

    Parameters
    ----------
    text : str
        Quote or page text.

    Returns
    -------
    str
        Uppercase, accent-free text with whitespace runs collapsed to a
        single space. Punctuation and digits are kept.
    """
    decomposed = unicodedata.normalize("NFKD", text.upper())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _quote_on_page(quote: str, page_text: str, threshold: float) -> tuple[bool, float]:
    """Check whether a quote occurs on a page.

    Parameters
    ----------
    quote : str
        Evidence quote.
    page_text : str
        Page text.
    threshold : float
        Minimum ``fuzz.partial_ratio`` accepted.

    Returns
    -------
    tuple of (bool, float)
        Whether the quote is grounded and the score (100.0 for an exact
        normalized substring).
    """
    norm_quote = normalize_for_grounding(quote)
    norm_page = normalize_for_grounding(page_text)
    if not norm_quote or not norm_page:
        return False, 0.0
    if norm_quote in norm_page:
        return True, 100.0
    score = float(fuzz.partial_ratio(norm_quote, norm_page))
    return score >= threshold, score


def _check_evidence(
    evidence: Evidence, page_texts: dict[int, str], threshold: float
) -> tuple[bool, str]:
    """Run the grounding check on one evidence object and mutate it.

    Parameters
    ----------
    evidence : Evidence
        Evidence to verify (mutated).
    page_texts : dict of int to str
        Page number to page text.
    threshold : float
        Minimum ``fuzz.partial_ratio`` accepted.

    Returns
    -------
    tuple of (bool, str)
        Whether the check passed and a short explanation on failure.
    """
    page_text = page_texts.get(evidence.page_number)
    if page_text is None:
        passed, detail = False, f"page {evidence.page_number} not available"
    else:
        passed, score = _quote_on_page(evidence.source_quote, page_text, threshold)
        detail = f"page {evidence.page_number}, partial_ratio {score:.2f} < {threshold:.2f}"
    evidence.grounding_verified = passed
    if not passed:
        evidence.confidence_score = min(evidence.confidence_score, GROUNDING_FAILED_CONFIDENCE_CAP)
    return passed, detail


def _shorten(text: str, limit: int = 60) -> str:
    """Truncate text for human-readable failure messages.

    Parameters
    ----------
    text : str
        Text to shorten.
    limit : int, default 60
        Maximum length kept.

    Returns
    -------
    str
        Text truncated with ``...`` when longer than ``limit``.
    """
    return text if len(text) <= limit else f"{text[:limit]}..."


def _field(model: DeedModel, name: str) -> FieldWithEvidence[Any]:
    """Return a contract field by name.

    Parameters
    ----------
    model : DeedModel
        Deed model.
    name : str
        Contract field name.

    Returns
    -------
    FieldWithEvidence
        The field wrapper.
    """
    field: FieldWithEvidence[Any] = getattr(model, name)
    return field


def _list_items(field: FieldWithEvidence[Any]) -> Iterator[tuple[int, Evidence]]:
    """Yield ``(index, evidence)`` for the items of a list field.

    Parameters
    ----------
    field : FieldWithEvidence
        List-valued field.

    Yields
    ------
    tuple of (int, Evidence)
        Item index and its evidence.
    """
    items = field.value if isinstance(field.value, list) else []
    for index, item in enumerate(items):
        evidence: Optional[Evidence] = getattr(item, "evidence", None)
        if evidence is not None:
            yield index, evidence


def verify_grounding(
    model: DeedModel,
    page_texts: dict[int, str],
    *,
    threshold: float = 90.0,
    job_id: str = "-",
    doc_sha256: str = "-",
) -> list[str]:
    """Verify that every evidence quote exists on its cited page.

    Parameters
    ----------
    model : DeedModel
        Extraction to verify (evidence and review flags mutated in place).
    page_texts : dict of int to str
        Page number (1-indexed) to page text (native text or OCR text).
    threshold : float, default 90.0
        Minimum ``rapidfuzz.fuzz.partial_ratio`` when the normalized quote
        is not an exact substring (``Settings.grounding_partial_ratio_threshold``).
    job_id : str, default "-"
        Job identifier for logging.
    doc_sha256 : str, default "-"
        Document hash for logging.

    Returns
    -------
    list of str
        Human-readable failure descriptions (empty when all pass).

    Notes
    -----
    Checks the 8 top-level evidences, every list item evidence (socios,
    representantes_legales, poderes) and every amendment event evidence.
    A failure sets ``grounding_verified=False``, caps the confidence at
    0.40 and flags the parent field for review. Amendment events have no
    review flag, so their failures are only reported. Fields with status
    ``no_encontrado`` / ``no_aplicable`` and no evidence are skipped; an
    ``extraido`` field without evidence is flagged as ungrounded.
    """
    failures: list[str] = []
    checked = 0
    for name in CONTRACT_FIELDS:
        field = _field(model, name)
        if field.evidence is None:
            if field.status is FieldStatus.EXTRACTED:
                field.flag_for_review(MISSING_EVIDENCE_REASON)
                failures.append(f"{name}: extracted value has no evidence")
        else:
            checked += 1
            passed, detail = _check_evidence(field.evidence, page_texts, threshold)
            if not passed:
                field.flag_for_review(GROUNDING_FAILURE_REASON)
                failures.append(
                    f"{name}: quote '{_shorten(field.evidence.source_quote)}' not found ({detail})"
                )
        if name in LIST_FIELDS:
            for index, evidence in _list_items(field):
                checked += 1
                passed, detail = _check_evidence(evidence, page_texts, threshold)
                if not passed:
                    field.flag_for_review(GROUNDING_FAILURE_REASON)
                    failures.append(
                        f"{name}[{index}]: quote '{_shorten(evidence.source_quote)}' not found ({detail})"
                    )
    for index, event in enumerate(model.amendment_history):
        checked += 1
        passed, detail = _check_evidence(event.evidence, page_texts, threshold)
        if not passed:
            failures.append(
                f"amendment_history[{index}] ({event.event_type.value}): quote "
                f"'{_shorten(event.evidence.source_quote)}' not found ({detail})"
            )
    log_event(
        _LOGGER,
        tag=StatusTag.WARN if failures else StatusTag.OK,
        event="grounding_check",
        stage=_STAGE,
        job_id=job_id,
        doc_sha256=doc_sha256,
        details={"evidences_checked": checked, "failures": len(failures), "failure_details": failures},
    )
    return failures


def _below_thousand(number: int) -> str:
    """Spell an integer in ``[0, 999]`` in Spanish legal style.

    Parameters
    ----------
    number : int
        Value in ``[0, 999]``.

    Returns
    -------
    str
        Uppercase words without accents (empty string for 0).
    """
    if number == 100:
        return "CIEN"
    hundreds, rest = divmod(number, 100)
    words: list[str] = []
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if rest < 30:
        if rest:
            words.append(_UNITS[rest])
    else:
        tens, units = divmod(rest, 10)
        words.append(_TENS[tens] if units == 0 else f"{_TENS[tens]} Y {_UNITS[units]}")
    return " ".join(words)


def spanish_amount_to_words(amount: float) -> str:
    """Spell a peso amount in Mexican legal Spanish.

    Parameters
    ----------
    amount : float
        Amount in ``[0, 999,999,999]``; cents are ignored (integer part).

    Returns
    -------
    str
        Uppercase words without accents, for example ``1000000.0`` ->
        ``"UN MILLON"``, ``250000.0`` -> ``"DOSCIENTOS CINCUENTA MIL"``,
        ``21000.0`` -> ``"VEINTIUN MIL"``, ``0`` -> ``"CERO"``.

    Raises
    ------
    ValueError
        If the amount is negative, not finite or above 999,999,999.

    Notes
    -----
    Uses the apocopated ``UN`` / ``VEINTIUN`` forms (``"VEINTIUN MIL"``,
    ``"CIENTO UN PESOS"``) as written before ``PESOS`` in notarial deeds,
    and ``"MIL"`` (not ``"UN MIL"``) for 1,000.
    """
    if amount != amount or amount in (float("inf"), float("-inf")):
        raise ValueError("amount must be a finite number")
    integer = int(amount)
    if integer < 0 or integer > _MAX_AMOUNT:
        raise ValueError(f"amount must be between 0 and {_MAX_AMOUNT}")
    if integer == 0:
        return "CERO"
    millions, remainder = divmod(integer, 1_000_000)
    thousands, units = divmod(remainder, 1_000)
    words: list[str] = []
    if millions == 1:
        words.append("UN MILLON")
    elif millions > 1:
        words.append(f"{_below_thousand(millions)} MILLONES")
    if thousands == 1:
        words.append("MIL")
    elif thousands > 1:
        words.append(f"{_below_thousand(thousands)} MIL")
    if units:
        words.append(_below_thousand(units))
    return " ".join(words)


def _letter_matches(expected_words: str, letter: str) -> bool:
    """Check that an amount-in-words string starts with the expected words.

    Parameters
    ----------
    expected_words : str
        Output of :func:`spanish_amount_to_words`.
    letter : str
        Normalized ``monto_en_letra``.

    Returns
    -------
    bool
        ``True`` when ``letter`` begins with the expected words followed by
        a word boundary. ``"UN MIL ..."`` is accepted for ``"MIL ..."``.
    """
    candidates = [expected_words]
    if expected_words.startswith("MIL"):
        candidates.append(f"UN {expected_words}")
    for candidate in candidates:
        if letter == candidate or letter.startswith(f"{candidate} "):
            return True
    return False


def verify_capital_consistency(model: DeedModel, *, job_id: str = "-", doc_sha256: str = "-") -> bool:
    """Check that the numeric capital matches its amount in words.

    Parameters
    ----------
    model : DeedModel
        Extraction to verify (``capital_social_actual`` flagged on mismatch).
    job_id : str, default "-"
        Job identifier for logging.
    doc_sha256 : str, default "-"
        Document hash for logging.

    Returns
    -------
    bool
        ``True`` when consistent or when there is no capital value to
        check; ``False`` on mismatch.

    Notes
    -----
    The words for ``monto_numerico`` must be the leading phrase of the
    normalized ``monto_en_letra`` (``"DOSCIENTOS CINCUENTA MIL PESOS 00/100
    M.N."`` matches ``250000.0``). When the letter states cents as
    ``NN/100`` they must equal the numeric cents.
    """
    field = model.capital_social_actual
    capital = field.value
    if capital is None:
        return True
    letter = normalize_for_grounding(capital.monto_en_letra)
    try:
        expected = spanish_amount_to_words(capital.monto_numerico)
        consistent = _letter_matches(expected, letter)
    except ValueError:
        expected = ""
        consistent = False
    cents_match = re.search(r"\b(\d{2})/100\b", letter)
    if consistent and cents_match is not None:
        numeric_cents = int(round((capital.monto_numerico - int(capital.monto_numerico)) * 100))
        consistent = int(cents_match.group(1)) == numeric_cents
    if not consistent:
        field.flag_for_review(CAPITAL_MISMATCH_REASON)
    log_event(
        _LOGGER,
        tag=StatusTag.OK if consistent else StatusTag.REVIEW_REQUIRED,
        event="capital_consistency_check",
        stage=_STAGE,
        job_id=job_id,
        doc_sha256=doc_sha256,
        details={
            "monto_numerico": capital.monto_numerico,
            "expected_words": expected,
            "monto_en_letra": capital.monto_en_letra,
            "consistent": consistent,
        },
    )
    return consistent


def apply_confidence_routing(
    model: DeedModel,
    threshold: float = 0.80,
    *,
    preserve_existing_reason: bool = False,
    job_id: str = "-",
    doc_sha256: str = "-",
) -> list[str]:
    """Flag fields whose evidence confidence is below a threshold.

    Parameters
    ----------
    model : DeedModel
        Extraction to route (review flags mutated in place).
    threshold : float, default 0.80
        Minimum acceptable confidence (``Settings.review_confidence_threshold``).
    preserve_existing_reason : bool, default False
        When ``True``, a field already flagged for review keeps its
        existing reason unchanged (it is still reported). Useful when an
        upstream stage has written an exact, more specific reason.
    job_id : str, default "-"
        Job identifier for logging.
    doc_sha256 : str, default "-"
        Document hash for logging.

    Returns
    -------
    list of str
        Flagged locations, e.g. ``"capital_social_actual"`` or
        ``"socios[1]"``.

    Notes
    -----
    Reason format: ``"Low confidence score (0.56 < 0.80)"``. List item
    evidences flag their parent list field.
    """
    flagged: list[str] = []

    def _route(field: FieldWithEvidence[Any], location: str, evidence: Evidence) -> None:
        """Flag ``field`` when ``evidence`` is below the threshold.

        Parameters
        ----------
        field : FieldWithEvidence
            Field to flag.
        location : str
            Reported location name.
        evidence : Evidence
            Evidence whose confidence is checked.
        """
        if evidence.confidence_score >= threshold:
            return
        flagged.append(location)
        if preserve_existing_reason and field.requires_human_review:
            return
        field.flag_for_review(f"Low confidence score ({evidence.confidence_score:.2f} < {threshold:.2f})")

    for name in CONTRACT_FIELDS:
        field = _field(model, name)
        if field.evidence is not None:
            _route(field, name, field.evidence)
        if name in LIST_FIELDS:
            for index, evidence in _list_items(field):
                _route(field, f"{name}[{index}]", evidence)
    if flagged:
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="low_confidence_routing",
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            details={"threshold": threshold, "flagged": flagged},
        )
    return flagged


__all__ = [
    "CAPITAL_MISMATCH_REASON",
    "CONTRACT_FIELDS",
    "DeedModel",
    "GROUNDING_FAILED_CONFIDENCE_CAP",
    "GROUNDING_FAILURE_REASON",
    "LIST_FIELDS",
    "MISSING_EVIDENCE_REASON",
    "SCALAR_FIELDS",
    "apply_confidence_routing",
    "normalize_for_grounding",
    "spanish_amount_to_words",
    "verify_capital_consistency",
    "verify_grounding",
]
