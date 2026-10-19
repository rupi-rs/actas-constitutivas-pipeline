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
from collections.abc import Iterator, Mapping
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

STRICT_FUZZY_SCORE: float = 95.0
"""Fuzzy score from which one-character OCR misreads in long words are tolerated."""

MIN_CHECKED_WORD_LENGTH: int = 3
"""Minimum length of an alphabetic quote token that must appear in the page window."""

MIN_SUBSTITUTION_WORD_LENGTH: int = 5
"""Minimum length of a quote word that may differ from the page by one substituted character."""

GROUNDING_STOPWORDS: frozenset[str] = frozenset(
    {"DE", "DEL", "LA", "LAS", "LOS", "EL", "Y", "EN", "AL", "POR", "CON", "QUE", "SU", "SUS", "UN", "UNA"}
)
"""Spanish function words skipped by the grounding word guard."""

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
_WORD_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Z0-9]*[A-Z][A-Z0-9]*")
_STANDALONE_DIGITS_RE: re.Pattern[str] = re.compile(r"(?<![A-Z0-9])\d+(?![A-Z0-9])")
_NUMERIC_RUN_RE: re.Pattern[str] = re.compile(r"(?<![A-Z0-9])[0-9OIL][0-9OIL,.'\u2019]*(?<![,.'\u2019])(?![A-Z0-9])")
_THOUSANDS_GAP_RE: re.Pattern[str] = re.compile(r"(?<=\d)(?:[,.'\u2019] ?| )(?=\d{3}(?!\d))")
_DECIMAL_SEPARATOR_RE: re.Pattern[str] = re.compile(r"(?<=\d)[.,](?=\d)")
_CONFUSABLE_DIGITS: dict[int, str] = str.maketrans({"O": "0", "I": "1", "L": "1"})
_NUMBER_WORD_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Z]+|[^A-Z\s]+")
_AMOUNT_RE: re.Pattern[str] = re.compile(
    r"(?<!\d)\d{1,3}(?:(?:[,'\u2019.]| )\d{3})+(?:[.,]\d{1,2})?(?!\d)|(?<!\d)\d+(?:[.,]\d{1,2})?(?!\d)"
)
_AMOUNT_CENTS_RE: re.Pattern[str] = re.compile(r"[.,]\d{1,2}$")
_NUMERIC_DATE_RE: re.Pattern[str] = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)")
_ISO_DATE_RE: re.Pattern[str] = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DEED_VALUE_RE: re.Pattern[str] = re.compile(r"\d{1,3}(?:[,.' ]\d{3})+|\d+")
_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^A-Z0-9]")
_NOTARY_WORD_RE: re.Pattern[str] = re.compile(r"\bNOTARI[AO]S?\b")
_NOTARY_CLAUSE_END_RE: re.Pattern[str] = re.compile(r"\s(?:DEL?|EN|CON|PARA)\s|[,;:()\[\]\n]|\s-\s")
_NOTARY_CLAUSE_WINDOW: int = 60
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

_WORD_UNITS: dict[str, int] = {
    "UN": 1,
    "UNO": 1,
    "UNA": 1,
    "DOS": 2,
    "TRES": 3,
    "CUATRO": 4,
    "CINCO": 5,
    "SEIS": 6,
    "SIETE": 7,
    "OCHO": 8,
    "NUEVE": 9,
}
_WORD_TEENS: dict[str, int] = {
    "ONCE": 11,
    "DOCE": 12,
    "TRECE": 13,
    "CATORCE": 14,
    "QUINCE": 15,
    "DIECISEIS": 16,
    "DIECISIETE": 17,
    "DIECIOCHO": 18,
    "DIECINUEVE": 19,
    "VEINTIUN": 21,
    "VEINTIUNO": 21,
    "VEINTIUNA": 21,
    "VEINTIDOS": 22,
    "VEINTITRES": 23,
    "VEINTICUATRO": 24,
    "VEINTICINCO": 25,
    "VEINTISEIS": 26,
    "VEINTISIETE": 27,
    "VEINTIOCHO": 28,
    "VEINTINUEVE": 29,
}
_WORD_TENS: dict[str, int] = {
    "DIEZ": 10,
    "VEINTE": 20,
    "TREINTA": 30,
    "CUARENTA": 40,
    "CINCUENTA": 50,
    "SESENTA": 60,
    "SETENTA": 70,
    "OCHENTA": 80,
    "NOVENTA": 90,
}
_WORD_HUNDREDS: dict[str, int] = {
    "CIEN": 100,
    "CIENTO": 100,
    **{f"{stem}{ending}": value for stem, value in (
        ("DOSCIENT", 200),
        ("TRESCIENT", 300),
        ("CUATROCIENT", 400),
        ("QUINIENT", 500),
        ("SEISCIENT", 600),
        ("SETECIENT", 700),
        ("OCHOCIENT", 800),
        ("NOVECIENT", 900),
    ) for ending in ("OS", "AS")},
}
_ORDINAL_UNITS: dict[str, int] = {
    **{f"{stem}{ending}": value for stem, value in (
        ("PRIMER", 1),
        ("SEGUND", 2),
        ("TERCER", 3),
        ("CUART", 4),
        ("QUINT", 5),
        ("SEXT", 6),
        ("SEPTIM", 7),
        ("SETIM", 7),
        ("OCTAV", 8),
        ("NOVEN", 9),
    ) for ending in ("O", "A")},
    "PRIMER": 1,
    "TERCER": 3,
}
_ORDINAL_TENS: dict[str, int] = {
    "DECIMO": 10,
    "DECIMA": 10,
    "VIGESIMO": 20,
    "VIGESIMA": 20,
    "TRIGESIMO": 30,
    "TRIGESIMA": 30,
}
_ORDINAL_SPECIAL: dict[str, int] = {"UNDECIMO": 11, "UNDECIMA": 11, "DUODECIMO": 12, "DUODECIMA": 12}
_COMPACT_ORDINAL_RE: re.Pattern[str] = re.compile(r"^(DECIM|VIGESIM|TRIGESIM)[OA]([A-Z]+)$")
_STAGE_START, _STAGE_HUNDREDS, _STAGE_TENS, _STAGE_AFTER_Y, _STAGE_DONE = range(5)


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


def canonicalize_numbers(text: str) -> str:
    """Canonicalize numbers in normalized text for OCR-tolerant comparison.

    Parameters
    ----------
    text : str
        Text already passed through :func:`normalize_for_grounding`.

    Returns
    -------
    str
        Text where, inside numeric runs that contain at least one digit,
        the OCR confusables ``O`` -> ``0`` and ``I`` / ``L`` -> ``1`` are
        replaced, and thousands / decimal separators (comma, period,
        apostrophe, a comma followed by a space, or a space before a
        three-digit group) are removed (``"$1,OOO,000.00"`` ->
        ``"$100000000"``, ``"30, 100"`` -> ``"30100"``, ``"1O"`` -> ``"10"``).

    Notes
    -----
    The same transformation is applied to the quote and the page, so
    canonicalization never makes two different numbers equal: the
    reviewer's altered amounts, notary numbers, deed numbers and years
    still differ after it.
    """

    def _fix(match: re.Match[str]) -> str:
        """Translate confusables in a numeric run that contains a digit.

        Parameters
        ----------
        match : re.Match
            Candidate numeric run.

        Returns
        -------
        str
            The translated run, or the run unchanged when it has no digit.
        """
        run = match.group(0)
        return run.translate(_CONFUSABLE_DIGITS) if any(char.isdigit() for char in run) else run

    fixed = _NUMERIC_RUN_RE.sub(_fix, text)
    fixed = _THOUSANDS_GAP_RE.sub("", fixed)
    return _DECIMAL_SEPARATOR_RE.sub("", fixed)


def _fuzzy_window_supports(norm_quote: str, window: str, score: float) -> bool:
    """Apply the content guards of the fuzzy grounding path.

    Parameters
    ----------
    norm_quote : str
        Normalized, number-canonicalized quote.
    window : str
        Normalized, number-canonicalized best-aligned page window.
    score : float
        ``partial_ratio`` of the quote against the page.

    Returns
    -------
    bool
        ``True`` when both guards pass (see Notes).

    Notes
    -----
    Guard (a), numbers: the multiset of standalone digit groups of the
    quote (see :func:`canonicalize_numbers`; digits embedded in words such
    as ``"L0PEZ"`` are left to the word guard) must be contained in the
    digit groups of the window. It rejects altered amounts, dates and
    numbers that still score above 90 (``"$7,000,000.00"`` against
    ``"$1,000,000.00"``, ``"Numero 51"`` against ``"Numero 15"``,
    ``"2018"`` against ``"2019"``).

    Guard (b), words: every quote word of length >=
    :data:`MIN_CHECKED_WORD_LENGTH` that is not in
    :data:`GROUNDING_STOPWORDS` must occur in the window with whitespace
    and punctuation removed. This rejects altered names (``"PEREZ
    LOZANO"`` against ``"PEREZ LOPEZ"``, ``"PEREZ PAZ"`` against ``"PEREZ
    GIL"``) while tolerating OCR whitespace and punctuation changes
    (``"PEREZLOPEZ"``) and accents. Only when the score is at least
    :data:`STRICT_FUZZY_SCORE`, each missing word of length >=
    :data:`MIN_SUBSTITUTION_WORD_LENGTH` may differ from the window by one
    substituted character (OCR misreads such as ``"ORTLZ"`` or
    ``"L0PEZ"``). Anything else is routed to human review.
    """
    quote_digits = Counter(_STANDALONE_DIGITS_RE.findall(norm_quote))
    window_digits = Counter(_STANDALONE_DIGITS_RE.findall(window))
    if quote_digits - window_digits:
        return False
    compact_window = _NON_ALNUM_RE.sub("", window)
    words = [
        word
        for word in _WORD_TOKEN_RE.findall(norm_quote)
        if len(word) >= MIN_CHECKED_WORD_LENGTH and word not in GROUNDING_STOPWORDS
    ]
    missing = [word for word in words if word not in compact_window]
    if not missing:
        return True
    if score < STRICT_FUZZY_SCORE:
        return False
    return all(
        len(word) >= MIN_SUBSTITUTION_WORD_LENGTH and _within_one_substitution(word, compact_window)
        for word in missing
    )


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
    Deliberate hardening beyond spec 03. An exact normalized substring
    passes, then an exact substring after :func:`canonicalize_numbers` on
    both sides. Otherwise the fuzzy path (``partial_ratio >= threshold``
    on the canonicalized texts) additionally requires
    :func:`_fuzzy_window_supports` on the best-aligned page window found by
    ``rapidfuzz.fuzz.partial_ratio_alignment`` (widened to word bounds).
    """
    norm_quote = normalize_for_grounding(quote)
    norm_page = normalize_for_grounding(page_text)
    if not norm_quote or not norm_page:
        return False, 0.0
    if norm_quote in norm_page:
        return True, 100.0
    canon_quote = canonicalize_numbers(norm_quote)
    canon_page = canonicalize_numbers(norm_page)
    if canon_quote in canon_page:
        return True, 100.0
    alignment = fuzz.partial_ratio_alignment(canon_quote, canon_page)
    if alignment is None:
        return False, 0.0
    score = float(alignment.score)
    if score < threshold:
        return False, score
    window = _expand_to_word_bounds(canon_page, alignment.dest_start, alignment.dest_end)
    return _fuzzy_window_supports(canon_quote, window, score), score


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
    it drives (each ``revoked_appointment_evidence`` entry of an event is
    grounded too and a failure flags ``representantes_legales``): ``REVOCATION`` and ``GRANT_OF_POWERS`` flag
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
        for sub_index, appointment in enumerate(event.revoked_appointment_evidence):
            checked += 1
            passed, detail = _check_evidence(appointment, page_texts, threshold)
            if not passed:
                model.representantes_legales.flag_for_review(
                    f"Grounding check failed for revoked appointment evidence of {event.event_type.value} "
                    f"event of deed {event.numero_escritura}: quote not found on specified page"
                )
                failures.append(
                    f"amendment_history[{index}].revoked_appointment_evidence[{sub_index}]: quote "
                    f"'{_shorten(appointment.source_quote)}' not found ({detail})"
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


def _parse_number_at(tokens: list[str], start: int) -> tuple[int, int]:
    """Parse the longest Spanish cardinal number starting at a token.

    Parameters
    ----------
    tokens : list of str
        Uppercase, accent-free tokens (punctuation tokens break numbers).
    start : int
        Index of the first token.

    Returns
    -------
    tuple of (int, int)
        Parsed value and number of tokens consumed (0 when no number
        starts at ``start``).

    Notes
    -----
    Each group below one thousand follows the order hundreds, tens,
    ``Y``, units (``"CIENTO TREINTA Y UNO"``, ``"DIEZ Y SEIS"``); ``MIL``
    and ``MILLON`` / ``MILLONES`` close a group. An out-of-order token ends
    the number.
    """
    if tokens[start] == "CERO":
        return 0, 1
    total = 0
    group = 0
    stage = _STAGE_START
    thousands_seen = False
    millions_seen = False
    last_value = 0
    consumed = 0
    index = start
    while index < len(tokens):
        token = tokens[index]
        if token == "Y":
            if stage == _STAGE_TENS and index + 1 < len(tokens) and tokens[index + 1] in _WORD_UNITS:
                stage = _STAGE_AFTER_Y
                index += 1
                continue
            break
        if token in _WORD_UNITS:
            if stage not in (_STAGE_START, _STAGE_HUNDREDS, _STAGE_AFTER_Y):
                break
            group += _WORD_UNITS[token]
            stage = _STAGE_DONE
        elif token in _WORD_TEENS:
            if stage not in (_STAGE_START, _STAGE_HUNDREDS):
                break
            group += _WORD_TEENS[token]
            stage = _STAGE_DONE
        elif token in _WORD_TENS:
            if stage not in (_STAGE_START, _STAGE_HUNDREDS):
                break
            group += _WORD_TENS[token]
            stage = _STAGE_TENS
        elif token in _WORD_HUNDREDS:
            if stage != _STAGE_START or group:
                break
            group += _WORD_HUNDREDS[token]
            stage = _STAGE_HUNDREDS
        elif token == "MIL":
            if thousands_seen:
                break
            total += (group or 1) * 1_000
            group = 0
            stage = _STAGE_START
            thousands_seen = True
        elif token in ("MILLON", "MILLONES"):
            if millions_seen or thousands_seen or not group:
                break
            total += group * 1_000_000
            group = 0
            stage = _STAGE_START
            millions_seen = True
        else:
            break
        index += 1
        last_value = total + group
        consumed = index - start
    return last_value, consumed


def spanish_numbers_in_text(text: str) -> list[int]:
    """Return every Spanish cardinal number written in words in a text.

    Parameters
    ----------
    text : str
        Any text; it is normalized with :func:`normalize_for_grounding`.

    Returns
    -------
    list of int
        Values in ``[0, 999,999,999]`` in order of appearance
        (``"DOS MIL VEINTIUNO"`` -> ``[2021]``, ``"TREINTA MIL CIEN"`` ->
        ``[30100]``, ``"UN MILLON DE PESOS"`` -> ``[1000000]``).

    Notes
    -----
    Supports ``UN`` / ``UNO`` / ``UNA``, ``VEINTIUNO`` .. ``VEINTINUEVE``,
    ``"DIEZ Y SEIS"`` style teens, ``"TREINTA Y UNO"``, ``MIL`` and
    ``MILLON`` / ``MILLONES``. Punctuation and digits break numbers.
    """
    tokens = _NUMBER_WORD_TOKEN_RE.findall(normalize_for_grounding(text))
    values: list[int] = []
    index = 0
    while index < len(tokens):
        value, consumed = _parse_number_at(tokens, index)
        if consumed:
            values.append(value)
            index += consumed
        else:
            index += 1
    return values


def spanish_words_to_int(phrase: str) -> Optional[int]:
    """Parse a phrase that is exactly one Spanish cardinal number.

    Parameters
    ----------
    phrase : str
        Phrase such as ``"CUARENTA Y CINCO MIL CIENTO VEINTE"``.

    Returns
    -------
    int or None
        The value, or ``None`` when the phrase is not one complete number.
    """
    tokens = _NUMBER_WORD_TOKEN_RE.findall(normalize_for_grounding(phrase))
    if not tokens:
        return None
    value, consumed = _parse_number_at(tokens, 0)
    return value if consumed == len(tokens) else None


def spanish_ordinals_in_text(text: str) -> list[int]:
    """Return the Spanish ordinal numbers (1 to 39) written in a text.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    list of int
        Values such as ``"PRIMERO"`` -> 1, ``"DECIMO QUINTO"`` -> 15,
        ``"DECIMOQUINTA"`` -> 15, ``"VIGESIMO"`` -> 20, ``"UNDECIMO"`` -> 11.
    """
    tokens = _NUMBER_WORD_TOKEN_RE.findall(normalize_for_grounding(text))
    values: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        compact = _COMPACT_ORDINAL_RE.match(token)
        if token in _ORDINAL_TENS:
            value = _ORDINAL_TENS[token]
            if index + 1 < len(tokens) and tokens[index + 1] in _ORDINAL_UNITS:
                value += _ORDINAL_UNITS[tokens[index + 1]]
                index += 1
            values.append(value)
        elif compact is not None and compact.group(2) in _ORDINAL_UNITS:
            values.append(_ORDINAL_TENS[f"{compact.group(1)}O"] + _ORDINAL_UNITS[compact.group(2)])
        elif token in _ORDINAL_SPECIAL:
            values.append(_ORDINAL_SPECIAL[token])
        elif token in _ORDINAL_UNITS:
            values.append(_ORDINAL_UNITS[token])
        index += 1
    return values


def _grouped_integers(norm_quote: str) -> list[int]:
    """Return the integer parts of the numbers of a quote.

    Parameters
    ----------
    norm_quote : str
        Normalized quote.

    Returns
    -------
    list of int
        Integer part of every number, accepting comma, period, apostrophe
        or space thousands separators and a 1-2 digit decimal part
        (``"$1'000,000.00"`` and ``"$1 000 000.00"`` -> ``1000000``,
        ``"30 100"`` -> ``30100``).
    """
    values: list[int] = []
    for token in _AMOUNT_RE.findall(norm_quote):
        integer = re.sub(r"[^\d]", "", _AMOUNT_CENTS_RE.sub("", token))
        if integer:
            values.append(int(integer))
    return values


def _number_in_quote(number: int, norm_quote: str, tokens: list[str], *, ordinals: bool = False) -> bool:
    """Check whether an integer appears in a quote as digits, words or an ordinal.

    Parameters
    ----------
    number : int
        Value to look for.
    norm_quote : str
        Normalized quote.
    tokens : list of str
        Digit groups of the quote (:func:`digit_tokens`).
    ordinals : bool, default False
        Whether Spanish ordinals (``"DECIMO QUINTO"``) also count.

    Returns
    -------
    bool
        ``True`` when a digit group equals the number (leading zeros
        ignored), the number is written in Spanish words, or, with
        ``ordinals``, written as a Spanish ordinal. ``"PRIMERO"`` always
        counts for 1 (``"primero de mayo"``).
    """
    if any(int(token) == number for token in tokens):
        return True
    if number in spanish_numbers_in_text(norm_quote):
        return True
    if ordinals:
        return number in spanish_ordinals_in_text(norm_quote)
    return number == 1 and re.search(r"\bPRIMERO\b", norm_quote) is not None


def notary_clause_numbers(text: str) -> list[int]:
    """Return the numbers stated in the notary-number clauses of a text.

    Parameters
    ----------
    text : str
        Quote or page text; it is normalized with :func:`normalize_for_grounding`.

    Returns
    -------
    list of int
        For every ``NOTARIA`` / ``NOTARIO`` word, the numbers written in
        digits, Spanish words or Spanish ordinals between it and the end of
        the clause (the first ``DE`` / ``DEL`` / ``EN``, comma, semicolon,
        colon, parenthesis or spaced dash, within 60 characters), in order
        (``"Notaria Publica Numero 42 del Primer Distrito"`` -> ``[42]``,
        ``"Notario Publico numero QUINCE de Guadalajara"`` -> ``[15]``).
    """
    norm = normalize_for_grounding(text)
    values: list[int] = []
    for match in _NOTARY_WORD_RE.finditer(norm):
        window = norm[match.end() : match.end() + _NOTARY_CLAUSE_WINDOW]
        end = _NOTARY_CLAUSE_END_RE.search(window)
        clause = window[: end.start()] if end is not None else window
        values.extend(int(token) for token in digit_tokens(clause) if token)
        values.extend(spanish_numbers_in_text(clause))
        values.extend(spanish_ordinals_in_text(clause))
    return values


def _notary_supported(
    number: int,
    norm_quote: str,
    tokens: list[str],
    page_text: Optional[str],
) -> tuple[bool, str]:
    """Check a notary number against its quote and, if needed, its page.

    Parameters
    ----------
    number : int
        Extracted ``numero_notaria``.
    norm_quote : str
        Normalized evidence quote.
    tokens : list of str
        Digit groups of the quote.
    page_text : str or None
        Text of the evidence page, when available.

    Returns
    -------
    tuple of (bool, str)
        Support verdict and its source: ``"quote"`` when the quote states
        the number, ``"quote_contradicts"`` when the quote holds a notary
        clause with other numbers, ``"page"`` when the quote covers only
        the notary name and a notary clause of the same page states the
        number, ``"none"`` otherwise.

    Notes
    -----
    Live extraction often quotes only the notary name (``"FERNANDO GARZA
    TREVINO"``) while the number sits in the adjacent clause or the running
    header of the same page. The number is required inside the quote only
    when the quote itself covers a notary-number clause; otherwise a notary
    clause on the cited page must state it. Grounding separately proves the
    quote is on that page. When a page names several notaries, a matching
    clause of any of them is accepted.
    """
    if _number_in_quote(number, norm_quote, tokens, ordinals=True):
        return True, "quote"
    if notary_clause_numbers(norm_quote):
        return False, "quote_contradicts"
    if page_text is not None and number in notary_clause_numbers(page_text):
        return True, "page"
    return False, "none"


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
        ``True`` when a numeric date (``10.05.2019``, ``10/05/2019``,
        ``10-05-2019`` or ``2019-05-10``) equals the value, or when the
        year, the day and the month (Spanish month name or month number)
        all appear in the quote; the year and day may be written in digits,
        words or (day) ordinals.
    """
    parsed = _parse_iso_date(value)
    if parsed is None:
        return False
    for day, month, year in _NUMERIC_DATE_RE.findall(norm_quote):
        if (int(year), int(month), int(day)) == (parsed.year, parsed.month, parsed.day):
            return True
    for year, month, day in _ISO_DATE_RE.findall(norm_quote):
        if (int(year), int(month), int(day)) == (parsed.year, parsed.month, parsed.day):
            return True
    month_ok = (
        re.search(rf"\b{_SPANISH_MONTHS[parsed.month - 1]}\b", norm_quote) is not None
        or (parsed.month == 9 and re.search(r"\bSETIEMBRE\b", norm_quote) is not None)
        or any(int(token) == parsed.month for token in tokens)
    )
    return (
        month_ok
        and _number_in_quote(parsed.year, norm_quote, tokens)
        and _number_in_quote(parsed.day, norm_quote, tokens, ordinals=True)
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
        Normalized quote.

    Returns
    -------
    bool
        ``True`` when a number in the quote has the same integer part
        (``"$1,000,000.00"``, ``"1000000"``, ``"$1'000,000.00"``,
        ``"$1 000 000.00"``, ``"$1.000.000,00"``) or the amount is written
        in Spanish words (``"UN MILLON DE PESOS"``).
    """
    expected = int(round(amount, 2))
    if str(expected) in tokens:
        return True
    if expected in _grouped_integers(norm_quote):
        return True
    return expected in spanish_numbers_in_text(norm_quote)


def _deed_number_supported(value: str, norm_quote: str, tokens: list[str]) -> bool:
    """Check that a deed number is supported by its quote.

    Parameters
    ----------
    value : str
        Deed number as extracted (``"30,100"``).
    norm_quote : str
        Normalized quote.
    tokens : list of str
        Digit groups of the quote.

    Returns
    -------
    bool
        ``True`` when every digit group of the value occurs among the digit
        groups of the quote, or when the value is a single number that
        occurs with any thousands separator (``"30 100"``) or in Spanish
        words (``"TREINTA MIL CIEN"``).
    """
    value_tokens = Counter(digit_tokens(value))
    if value_tokens and not value_tokens - Counter(tokens):
        return True
    stripped = value.strip()
    if not stripped or _DEED_VALUE_RE.fullmatch(stripped) is None:
        return False
    number = int(re.sub(r"[^\d]", "", stripped))
    return number in _grouped_integers(norm_quote) or number in spanish_numbers_in_text(norm_quote)


def verify_value_in_quote(
    model: DeedModel,
    *,
    page_texts: Optional[Mapping[int, str]] = None,
    job_id: str = "-",
    doc_sha256: str = "-",
) -> list[str]:
    """Check that scalar values are supported by their own evidence quote.

    Parameters
    ----------
    model : DeedModel
        Extraction to verify (review flags mutated in place).
    page_texts : mapping of int to str or None, default None
        Page texts keyed by page number; used only for the notary number
        when the quote covers just the notary name.
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

    - ``numero_escritura``: digits (any thousands separator) or Spanish
      words (:func:`_deed_number_supported`).
    - ``fecha_constitucion``: a numeric date or year, day and month in
      digits or words (:func:`_date_supported`).
    - ``notario``: ``numero_notaria`` as digits, Spanish words or an
      ordinal (``"DECIMO QUINTO"``) in the quote; when the quote holds no
      notary-number clause at all (only the name), a notary clause on the
      cited page must state the number (:func:`_notary_supported`). A
      quote whose own notary clause states another number fails.
    - ``capital_social_actual``: the integer part of ``monto_numerico``
      as a number with any thousands separator or in Spanish words.

    Only fields with status ``extraido``, a value and evidence are
    checked. Failures are flagged with :data:`VALUE_NOT_IN_QUOTE_REASON`.
    """
    failures: list[str] = []
    checks: list[tuple[str, bool]] = []
    notary_source: Optional[str] = None
    for name in ("numero_escritura", "fecha_constitucion", "notario", "capital_social_actual"):
        field = _field(model, name)
        if field.status is not FieldStatus.EXTRACTED or field.value is None or field.evidence is None:
            continue
        norm_quote = normalize_for_grounding(field.evidence.source_quote)
        tokens = digit_tokens(norm_quote)
        if name == "numero_escritura":
            supported = _deed_number_supported(str(field.value), norm_quote, tokens)
        elif name == "fecha_constitucion":
            supported = _date_supported(str(field.value), norm_quote, tokens)
        elif name == "notario":
            page_text = None if page_texts is None else page_texts.get(field.evidence.page_number)
            supported, notary_source = _notary_supported(
                int(field.value.numero_notaria), norm_quote, tokens, page_text
            )
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
        details={
            "fields_checked": [name for name, _ in checks],
            "unsupported": failures,
            "notary_support": notary_source,
        },
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
    "GROUNDING_STOPWORDS",
    "MIN_CHECKED_WORD_LENGTH",
    "MIN_SUBSTITUTION_WORD_LENGTH",
    "SCALAR_FIELDS",
    "STRICT_FUZZY_SCORE",
    "TEMPORAL_INCONSISTENCY_REASON",
    "VALUE_NOT_IN_QUOTE_REASON",
    "apply_confidence_routing",
    "canonicalize_numbers",
    "digit_tokens",
    "normalize_for_grounding",
    "spanish_amount_to_words",
    "spanish_numbers_in_text",
    "spanish_ordinals_in_text",
    "spanish_words_to_int",
    "verify_capital_consistency",
    "verify_grounding",
    "verify_temporal_consistency",
    "notary_clause_numbers",
    "verify_value_in_quote",
]
