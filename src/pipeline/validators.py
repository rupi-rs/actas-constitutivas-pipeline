"""Deterministic post-extraction validators.

Implements the page-level grounding check, the numeric-versus-words capital
consistency check and confidence threshold routing (spec 03 section 2),
plus two deliberate hardening checks beyond spec 03: a value-in-quote
check for scalar fields and a temporal consistency check for the
amendment history.
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
from collections import Counter
from collections.abc import Iterator
from datetime import date
from typing import Any, Optional, Union

from rapidfuzz import fuzz

from src.core.logger import StatusTag, get_logger, log_event
from src.schemas.deed_models import (
    AmendmentEvent,
    AmendmentType,
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

VALUE_NOT_IN_QUOTE_REASON: str = "Value not supported by source quote"
"""Review reason appended when a scalar value cannot be found in its own quote."""

TEMPORAL_INCONSISTENCY_REASON: str = "Temporal inconsistency: act dated before the incorporation date"
"""Review reason prefix appended when an act predates ``fecha_constitucion``."""

STRICT_FUZZY_SCORE: float = 97.0
"""Fuzzy score from which a single one-character OCR misread in a word is tolerated."""

MIN_CHECKED_WORD_LENGTH: int = 4
"""Minimum length of an alphabetic quote token that must appear in the page window."""

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
_DIGIT_TOKEN_RE: re.Pattern[str] = re.compile(r"\d+(?:[.,]\d+)*")
_ALPHA_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Z]+")
_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^A-Z0-9]")
_CENTS_SUFFIX_RE: re.Pattern[str] = re.compile(r"[.,]\d{2}$")
_SPANISH_MONTHS: tuple[str, ...] = (
    "ENERO",
    "FEBRERO",
    "MARZO",
    "ABRIL",
    "MAYO",
    "JUNIO",
    "JULIO",
    "AGOSTO",
    "SEPTIEMBRE",
    "OCTUBRE",
    "NOVIEMBRE",
    "DICIEMBRE",
)

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


def digit_tokens(text: str) -> list[str]:
    """Return the digit groups of a text with thousands / decimal separators removed.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    list of str
        One string of digits per number, in order (``"$1,000,000.00 y 15"``
        -> ``["100000000", "15"]``).
    """
    return [re.sub(r"[.,]", "", token) for token in _DIGIT_TOKEN_RE.findall(text)]


def _expand_to_word_bounds(text: str, start: int, end: int) -> str:
    """Widen ``text[start:end]`` so it neither starts nor ends inside a word.

    Parameters
    ----------
    text : str
        Haystack text.
    start, end : int
        Window bounds from the fuzzy alignment.

    Returns
    -------
    str
        The widened window.
    """
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end]


def _within_one_substitution(word: str, text: str) -> bool:
    """Check whether ``word`` occurs in ``text`` with at most one substituted character.

    Parameters
    ----------
    word : str
        Word to look for.
    text : str
        Compact haystack (letters and digits only).

    Returns
    -------
    bool
        ``True`` when some substring of ``text`` with the length of
        ``word`` differs from it in at most one position.
    """
    size = len(word)
    for start in range(len(text) - size + 1):
        mismatches = 0
        for left, right in zip(word, text[start : start + size]):
            if left != right:
                mismatches += 1
                if mismatches > 1:
                    break
        if mismatches <= 1:
            return True
    return False


def _fuzzy_window_supports(norm_quote: str, window: str, score: float) -> bool:
    """Apply the content guards of the fuzzy grounding path.

    Parameters
    ----------
    norm_quote : str
        Normalized quote.
    window : str
        Normalized best-aligned page window.
    score : float
        ``partial_ratio`` of the quote against the page.

    Returns
    -------
    bool
        ``True`` when both guards pass (see Notes).

    Notes
    -----
    Guard (a), numbers: the multiset of digit groups of the quote
    (separators removed) must be contained in the digit groups of the
    window. It rejects altered amounts, dates and numbers that still score
    above 90 (``"$7,000,000.00"`` against ``"$1,000,000.00"``,
    ``"Numero 51"`` against ``"Numero 15"``, ``"2018"`` against
    ``"2019"``).

    Guard (b), words: every alphabetic quote token of length >=
    :data:`MIN_CHECKED_WORD_LENGTH` must occur in the window with
    whitespace and punctuation removed. This rejects altered names
    (``"PEREZ LOZANO"`` against ``"PEREZ LOPEZ"``) while tolerating OCR
    whitespace and punctuation changes (``"PEREZLOPEZ"``) and accents
    (both sides are accent-free). Only when the score is at least
    :data:`STRICT_FUZZY_SCORE`, a single token may differ from the window
    by one substituted character (a typical OCR misread such as
    ``"ORTLZ"`` for ``"ORTIZ"``). Anything else is routed to human review,
    which is the safe outcome.
    """
    quote_digits = Counter(digit_tokens(norm_quote))
    window_digits = Counter(digit_tokens(window))
    if quote_digits - window_digits:
        return False
    compact_window = _NON_ALNUM_RE.sub("", window)
    words = [word for word in _ALPHA_TOKEN_RE.findall(norm_quote) if len(word) >= MIN_CHECKED_WORD_LENGTH]
    missing = [word for word in words if word not in compact_window]
    if not missing:
        return True
    if score < STRICT_FUZZY_SCORE or len(missing) > 1:
        return False
    return _within_one_substitution(missing[0], compact_window)


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

    Notes
    -----
    Deliberate hardening beyond spec 03: the fuzzy path
    (``partial_ratio >= threshold``) additionally requires
    :func:`_fuzzy_window_supports` on the best-aligned page window found by
    ``rapidfuzz.fuzz.partial_ratio_alignment`` (widened to word bounds).
    An exact normalized substring always passes.
    """
    norm_quote = normalize_for_grounding(quote)
    norm_page = normalize_for_grounding(page_text)
    if not norm_quote or not norm_page:
        return False, 0.0
    if norm_quote in norm_page:
        return True, 100.0
    alignment = fuzz.partial_ratio_alignment(norm_quote, norm_page)
    if alignment is None:
        return False, 0.0
    score = float(alignment.score)
    if score < threshold:
        return False, score
    window = _expand_to_word_bounds(norm_page, alignment.dest_start, alignment.dest_end)
    return _fuzzy_window_supports(norm_quote, window, score), score


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
        detail = (
            f"page {evidence.page_number}, partial_ratio {score:.2f} < {threshold:.2f}"
            if score < threshold
            else f"page {evidence.page_number}, partial_ratio {score:.2f} but quote digits or words "
            "differ from the page"
        )
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


def _dependent_fields(event: AmendmentEvent) -> tuple[str, ...]:
    """Return the contract fields whose resolved value depends on an event.

    Parameters
    ----------
    event : AmendmentEvent
        Amendment event.

    Returns
    -------
    tuple of str
        ``("representantes_legales", "poderes")`` for revocations and
        grants, ``("capital_social_actual",)`` for capital increases and
        an empty tuple for the original incorporation, whose contract
        fields carry their own evidence.
    """
    if event.event_type in (AmendmentType.REVOCATION, AmendmentType.GRANT_OF_POWERS):
        return ("representantes_legales", "poderes")
    if event.event_type is AmendmentType.CAPITAL_INCREASE:
        return ("capital_social_actual",)
    return ()


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
    review flag of their own, so a failing event flags the contract fields
    it drives: ``REVOCATION`` and ``GRANT_OF_POWERS`` flag
    ``representantes_legales`` and ``poderes``; ``CAPITAL_INCREASE`` flags
    ``capital_social_actual``. Fields with status ``no_encontrado`` /
    ``no_aplicable`` and no evidence are skipped; an ``extraido`` field
    without evidence is flagged as ungrounded. The fuzzy path is hardened
    beyond spec 03 (see :func:`_quote_on_page`).
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
            reason = (
                f"Grounding check failed for {event.event_type.value} event of deed "
                f"{event.numero_escritura}: quote not found on specified page"
            )
            for dependent in _dependent_fields(event):
                _field(model, dependent).flag_for_review(reason)
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


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` string.

    Parameters
    ----------
    value : str or None
        Date string.

    Returns
    -------
    datetime.date or None
        Parsed date, or ``None`` when missing or malformed.
    """
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def verify_temporal_consistency(model: DeedModel, *, job_id: str = "-", doc_sha256: str = "-") -> list[str]:
    """Flag acts and appointments dated before the incorporation date.

    Parameters
    ----------
    model : DeedModel
        Extraction to verify (review flags mutated in place).
    job_id : str, default "-"
        Job identifier for logging.
    doc_sha256 : str, default "-"
        Document hash for logging.

    Returns
    -------
    list of str
        Human-readable inconsistency descriptions (empty when consistent
        or when ``fecha_constitucion`` is missing or unparseable).

    Notes
    -----
    Deliberate hardening beyond spec 03. No corporate act can predate the
    company, so an event whose ``fecha_acto`` precedes
    ``fecha_constitucion`` is a date error (OCR or LLM) that could silently
    change the temporal resolution. Flags ``representantes_legales`` and
    ``poderes`` for ``REVOCATION`` / ``GRANT_OF_POWERS`` events,
    ``capital_social_actual`` for ``CAPITAL_INCREASE`` events and
    ``fecha_constitucion`` for an ``INCORPORATION`` event. A legal
    representative whose ``fecha_nombramiento`` precedes the incorporation
    flags ``representantes_legales``. Works on raw and resolved models.
    """
    incorporation = _parse_iso_date(model.fecha_constitucion.value)
    issues: list[str] = []
    if incorporation is not None:
        for event in model.amendment_history:
            event_date = _parse_iso_date(event.fecha_acto)
            if event_date is None or event_date >= incorporation:
                continue
            description = (
                f"{event.event_type.value} event of deed {event.numero_escritura} dated {event.fecha_acto} "
                f"precedes fecha_constitucion {model.fecha_constitucion.value}"
            )
            targets = _dependent_fields(event) or ("fecha_constitucion",)
            for name in targets:
                _field(model, name).flag_for_review(f"{TEMPORAL_INCONSISTENCY_REASON} ({description})")
            issues.append(description)
        for rep in model.representantes_legales.value or []:
            appointed = _parse_iso_date(rep.fecha_nombramiento)
            if appointed is None or appointed >= incorporation:
                continue
            description = (
                f"appointment of '{rep.nombre}' dated {rep.fecha_nombramiento} "
                f"precedes fecha_constitucion {model.fecha_constitucion.value}"
            )
            model.representantes_legales.flag_for_review(f"{TEMPORAL_INCONSISTENCY_REASON} ({description})")
            issues.append(description)
    log_event(
        _LOGGER,
        tag=StatusTag.REVIEW_REQUIRED if issues else StatusTag.OK,
        event="temporal_consistency_check",
        stage=_STAGE,
        job_id=job_id,
        doc_sha256=doc_sha256,
        details={"fecha_constitucion": model.fecha_constitucion.value, "issues": issues},
    )
    return issues


def _number_in_quote(number: int, norm_quote: str, tokens: list[str]) -> bool:
    """Check whether an integer appears in a quote as digits or Spanish words.

    Parameters
    ----------
    number : int
        Value to look for.
    norm_quote : str
        Normalized quote.
    tokens : list of str
        Digit groups of the quote (:func:`digit_tokens`).

    Returns
    -------
    bool
        ``True`` when a digit group equals the number (leading zeros
        ignored) or its Spanish words occur as a whole phrase.
    """
    if any(int(token) == number for token in tokens):
        return True
    try:
        words = spanish_amount_to_words(float(number))
    except ValueError:
        return False
    variants = [words, "PRIMERO"] if number == 1 else [words]
    return any(re.search(rf"\b{re.escape(variant)}\b", norm_quote) for variant in variants)


def _date_supported(value: str, norm_quote: str, tokens: list[str]) -> bool:
    """Check that an ISO date is supported by its quote.

    Parameters
    ----------
    value : str
        Date ``YYYY-MM-DD``.
    norm_quote : str
        Normalized quote.
    tokens : list of str
        Digit groups of the quote.

    Returns
    -------
    bool
        ``True`` when the year, the day and the month (Spanish month name
        or month number) all appear in the quote; the year and day may be
        written in digits or words.
    """
    parsed = _parse_iso_date(value)
    if parsed is None:
        return False
    month_ok = (
        re.search(rf"\b{_SPANISH_MONTHS[parsed.month - 1]}\b", norm_quote) is not None
        or (parsed.month == 9 and re.search(r"\bSETIEMBRE\b", norm_quote) is not None)
        or any(int(token) == parsed.month for token in tokens)
    )
    return (
        month_ok
        and _number_in_quote(parsed.year, norm_quote, tokens)
        and _number_in_quote(parsed.day, norm_quote, tokens)
    )


def _amount_supported(amount: float, tokens: list[str], norm_quote: str) -> bool:
    """Check that a peso amount appears in its quote.

    Parameters
    ----------
    amount : float
        Numeric amount.
    tokens : list of str
        Digit groups of the quote (separators removed).
    norm_quote : str
        Normalized quote (used to drop cents from digit groups).

    Returns
    -------
    bool
        ``True`` when a number in the quote has the same integer part,
        e.g. ``1000000.0`` against ``"$1,000,000.00"`` or ``"1000000"``.
    """
    expected = str(int(round(amount, 2)))
    if expected in tokens:
        return True
    integer_parts = [
        re.sub(r"[.,]", "", _CENTS_SUFFIX_RE.sub("", token)) for token in _DIGIT_TOKEN_RE.findall(norm_quote)
    ]
    return expected in integer_parts


def verify_value_in_quote(model: DeedModel, *, job_id: str = "-", doc_sha256: str = "-") -> list[str]:
    """Check that scalar values are supported by their own evidence quote.

    Parameters
    ----------
    model : DeedModel
        Extraction to verify (review flags mutated in place).
    job_id : str, default "-"
        Job identifier for logging.
    doc_sha256 : str, default "-"
        Document hash for logging.

    Returns
    -------
    list of str
        Names of the fields whose value is not supported by the quote.

    Notes
    -----
    Deliberate hardening beyond spec 03: grounding proves the quote is on
    the page, this check proves the value is in the quote.

    - ``numero_escritura``: every digit group of the value occurs among
      the digit groups of the quote (separators ignored).
    - ``fecha_constitucion``: year, day and month occur in the quote.
    - ``notario``: ``numero_notaria`` occurs as digits or Spanish words.
    - ``capital_social_actual``: the integer part of ``monto_numerico``
      occurs as a number (``1,000,000.00`` or ``1000000``).

    Only fields with status ``extraido``, a value and evidence are
    checked. Failures are flagged with :data:`VALUE_NOT_IN_QUOTE_REASON`.
    """
    failures: list[str] = []
    checks: list[tuple[str, bool]] = []
    for name in ("numero_escritura", "fecha_constitucion", "notario", "capital_social_actual"):
        field = _field(model, name)
        if field.status is not FieldStatus.EXTRACTED or field.value is None or field.evidence is None:
            continue
        norm_quote = normalize_for_grounding(field.evidence.source_quote)
        tokens = digit_tokens(norm_quote)
        if name == "numero_escritura":
            value_tokens = Counter(digit_tokens(str(field.value)))
            supported = bool(value_tokens) and not value_tokens - Counter(tokens)
        elif name == "fecha_constitucion":
            supported = _date_supported(str(field.value), norm_quote, tokens)
        elif name == "notario":
            supported = _number_in_quote(int(field.value.numero_notaria), norm_quote, tokens)
        else:
            supported = _amount_supported(float(field.value.monto_numerico), tokens, norm_quote)
        checks.append((name, supported))
        if not supported:
            field.flag_for_review(VALUE_NOT_IN_QUOTE_REASON)
            failures.append(name)
    log_event(
        _LOGGER,
        tag=StatusTag.REVIEW_REQUIRED if failures else StatusTag.OK,
        event="value_in_quote_check",
        stage=_STAGE,
        job_id=job_id,
        doc_sha256=doc_sha256,
        details={"fields_checked": [name for name, _ in checks], "unsupported": failures},
    )
    return failures


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
    "MIN_CHECKED_WORD_LENGTH",
    "SCALAR_FIELDS",
    "STRICT_FUZZY_SCORE",
    "TEMPORAL_INCONSISTENCY_REASON",
    "VALUE_NOT_IN_QUOTE_REASON",
    "apply_confidence_routing",
    "digit_tokens",
    "normalize_for_grounding",
    "spanish_amount_to_words",
    "verify_capital_consistency",
    "verify_grounding",
    "verify_temporal_consistency",
    "verify_value_in_quote",
]
