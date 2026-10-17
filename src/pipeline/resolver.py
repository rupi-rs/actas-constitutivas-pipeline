"""Temporal entity resolution for multi-act corporate deed files.

Resolves the *current* legal state of a company from a pre-resolution
:class:`~src.schemas.deed_models.RawDeedExtraction` that lists every act
ever recorded in the file (original incorporation, capital increases,
revocations and new grants of powers).

Notes
-----
Resolution rules (CLAUDE.md section 5 rule 4, spec 03 section 1):

- ``amendment_history`` is sorted chronologically by ``fecha_acto`` with a
  stable sort, so same-day acts keep their document order.
- ``capital_social_actual`` is the latest-dated amount among the
  incorporation capital and every ``CAPITAL_INCREASE`` event.
- A legal representative is revoked when a ``REVOCATION`` event's
  ``sujeto_afectado`` matches the representative's normalized name with a
  similarity at or above ``Settings.fuzzy_match_threshold`` **and** the
  surnames and first given name agree **and** the appointment date is on or before the
  revocation date. Revoked representatives and their powers are removed
  from the active lists; the evidence of both acts is retained in the
  ``REVOCATION`` event.
- Ambiguous matches (high score with conflicting surnames, or a score in
  ``[REVIEW_SCORE_FLOOR, threshold)`` with agreeing surnames) never revoke
  silently and never keep silently: the representative stays listed and
  ``representantes_legales`` is flagged for human review with a reason.
- A matched revocation dated before the only appointment of the person is
  not applied but flags ``representantes_legales`` and ``poderes`` for
  review (``revocation_date_inconsistent``).
- A ``sujeto_afectado`` naming several people is split and every person
  is evaluated separately; the appointment evidence of each revoked person
  is copied into ``revoked_appointment_evidence`` of the revocation event.
  The unsplit subject is evaluated first, so a representative whose name
  contains ``" Y "`` is not split apart.
- Every subject of a ``REVOCATION`` or ``GRANT_OF_POWERS`` event must be
  named by that event's own ``source_quote``; an unsupported revocation
  subject is not applied and flags ``representantes_legales`` and
  ``poderes`` (``revocation_subject_unsupported``).
- Powers granted to a role (``"EL ADMINISTRADOR UNICO"``) follow the
  revoked holder of that role, and any active power whose grantee is no
  active representative (by name or role) flags ``poderes``.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from rapidfuzz import fuzz

from src.core.config import Settings
from src.core.logger import StatusTag, get_logger, log_event
from src.schemas.deed_models import (
    AmendmentEvent,
    AmendmentType,
    CapitalInfo,
    FieldStatus,
    LegalRepresentativeItem,
    PowerOfAttorneyItem,
    RawDeedExtraction,
)

_LOGGER = get_logger("pipeline.resolver")
_STAGE: str = "resolution"

REVIEW_SCORE_FLOOR: float = 75.0
"""Lower bound of the similarity band that triggers human review instead of a decision."""

INITIAL_EXPANSION_SCORE: float = 88.0
"""Minimum score granted when an abbreviated initial expands to a full given name."""

_PREFIX_ALTERNATION: str = (
    r"CONTADORA\s+PUBLICA|CONTADOR\s+PUBLICO|C\s*\.?\s*P|"
    r"EL\s+SENOR|LA\s+SENORA|LOS\s+SENORES|LAS\s+SENORAS|SENORES|SENORAS|SRES|SRAS|"
    "LICENCIADO|LICENCIADA|LICDO|LICDA|LIC|INGENIERO|INGENIERA|ING|"
    "ARQUITECTO|ARQUITECTA|ARQ|SENORITA|SENOR|SENORA|SRITA|SRA|SR|DONA|DON|"
    "DOCTOR|DOCTORA|DRA|DR|MAESTRO|MAESTRA|MTRO|MTRA|CIUDADANO|CIUDADANA|C"
)
"""Leading honorifics; multi-word and longer forms precede their prefixes so they win the alternation."""

_ARTICLE_ALTERNATION: str = "EL|LA|LOS|LAS|DEL|AL|Y"
"""Articles and conjunctions that may precede an honorific (``"EL SR."``, ``"LIC. Y C.P."``)."""

_LEADING_PREFIX_RE: re.Pattern[str] = re.compile(
    rf"^(?:(?:(?:{_ARTICLE_ALTERNATION})\s+)?(?:{_PREFIX_ALTERNATION})\b\.?\s*)+"
)
_SUBJECT_SEPARATOR_RE: re.Pattern[str] = re.compile(r"\s*[,;]\s*|\s+Y\s+|\s+E\s+(?=H?I)")
_LIST_SEPARATOR_RE: re.Pattern[str] = re.compile(r"\s*[,;]\s*")
_ENTITY_SUFFIX_PATTERN: str = (
    r"(?<![A-Z])(?:S\.?\s*A\.?\s*(?:P\.?\s*I\.?\s*)?(?:B\.?\s*)?|S\.?\s*DE\s+R\.?\s*L\.?\s*|"
    r"S\.?\s*C\.?\s*|A\.?\s*C\.?\s*)(?:DE\s+C\.?\s*V\.?)?(?![A-Z])"
)
"""Mexican legal-entity suffixes (S.A., S.A.P.I., S.A.B., S. DE R.L., S.C., A.C., optionally DE C.V.)."""

_ENTITY_SUFFIX_RE: re.Pattern[str] = re.compile(_ENTITY_SUFFIX_PATTERN)
_JOB_TITLE_RE: re.Pattern[str] = re.compile(
    r"^(?:(?:EL|LA)\s+)?(?:ADMINISTRADORA?\s+UNIC[OA]|ADMINISTRADORA?|APODERAD[OA](?:\s+(?:GENERAL|LEGAL|ESPECIAL))?|"
    r"GERENTE(?:\s+GENERAL)?|DIRECTORA?(?:\s+GENERAL)?|PRESIDENTE|PRESIDENTA|SECRETARI[OA]|COMISARI[OA])\.?$"
)
"""Job-title fragments dropped from a revocation subject (``"..., ADMINISTRADORA UNICA"``)."""

_ROLE_ARTICLE_RE: re.Pattern[str] = re.compile(r"^(?:(?:EL|LA|LOS|LAS|AL|DEL|A)\s+)+")
_PUNCTUATION_RE: re.Pattern[str] = re.compile(r"[.,;\-]")
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
_PARENTHESIS_RE: re.Pattern[str] = re.compile(r"\(([^()]*)\)")


class MatchDecision(str, Enum):
    """Outcome of comparing a revocation subject with a candidate person.

    Notes
    -----
    ``MATCH`` revokes, ``REVIEW`` keeps the person but requires human
    review, ``NO_MATCH`` leaves the person untouched.
    """

    MATCH = "match"
    REVIEW = "review"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class PersonMatch:
    """Scored comparison between two normalized person names.

    Parameters
    ----------
    score : float
        Similarity in ``[0, 100]`` from :func:`person_similarity`.
    surnames_agree : bool
        Result of :func:`surnames_agree`.
    given_name_agrees : bool
        Result of :func:`first_given_name_agrees`.
    decision : MatchDecision
        Resulting decision.
    """

    score: float
    surnames_agree: bool
    given_name_agrees: bool
    decision: MatchDecision


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks from a string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text after NFKD decomposition without combining characters
        (``"PÉREZ"`` -> ``"PEREZ"``, ``"Ñ"`` -> ``"N"``).
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_person_name(raw_name: str) -> str:
    """Normalize a Mexican person name for entity matching.

    Parameters
    ----------
    raw_name : str
        Name as written in the deed (for example
        ``"LIC. JUAN CARLOS PÉREZ LÓPEZ"``).

    Returns
    -------
    str
        Uppercase, accent-free, honorific-free name with single spaces
        (``"JUAN CARLOS PEREZ LOPEZ"``).

    Notes
    -----
    Steps (spec 03 section 1):

    1. Uppercase and strip surrounding whitespace.
    2. NFKD accent stripping.
    3. Removal of professional / courtesy prefixes. The prefix list
       contains ``C`` (Ciudadano), so the regex is anchored at the start of
       the name and applied as a repeated group: only leading honorifics
       (possibly several, e.g. ``"C. LIC. ..."``) are removed. A middle
       initial such as the ``C.`` in ``"JUAN C. PEREZ LOPEZ"`` is kept,
       which is what makes initial-expansion matching possible. If
       stripping would leave nothing, the unstripped text is kept. The
       list extends spec 03 with ``C.P.`` / ``CP`` / ``CONTADOR PUBLICO``,
       ``LICDO`` / ``LICDA``, ``ARQ`` / ``ARQUITECTO``, ``DON`` / ``DONA``
       and ``EL SENOR`` / ``LA SENORA`` (matched after accent stripping).
    4. Punctuation ``. , ; -`` is replaced by spaces and runs of whitespace
       are collapsed.
    """
    text = strip_accents(raw_name.upper().strip())
    stripped = _LEADING_PREFIX_RE.sub("", text)
    if stripped.strip():
        text = stripped
    text = _PUNCTUATION_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _initials_expand(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """Check whether two token lists match under initial expansion.

    Parameters
    ----------
    tokens_a, tokens_b : list of str
        Tokens of two normalized names.

    Returns
    -------
    bool
        ``True`` when both lists have the same length (at least 3), the
        first token and the last two tokens are identical, every token
        pair is identical or a single-letter initial of the other token,
        and at least one pair is an initial expansion.
    """
    if len(tokens_a) != len(tokens_b) or len(tokens_a) < 3:
        return False
    if tokens_a[0] != tokens_b[0] or tokens_a[-2:] != tokens_b[-2:]:
        return False
    expanded = False
    for left, right in zip(tokens_a, tokens_b):
        if left == right:
            continue
        if len(left) == 1 and right.startswith(left):
            expanded = True
            continue
        if len(right) == 1 and left.startswith(right):
            expanded = True
            continue
        return False
    return expanded


def person_similarity(a: str, b: str) -> float:
    """Score the similarity of two normalized person names.

    Parameters
    ----------
    a, b : str
        Names already passed through :func:`normalize_person_name`.

    Returns
    -------
    float
        ``max(token_sort_ratio, token_set_ratio * 0.95)`` in ``[0, 100]``,
        raised to at least :data:`INITIAL_EXPANSION_SCORE` when the names
        match under initial expansion (``"JUAN C PEREZ LOPEZ"`` vs
        ``"JUAN CARLOS PEREZ LOPEZ"``).

    Notes
    -----
    The score alone is not a revocation decision; see
    :func:`classify_person_match`, which adds a surname guard.
    """
    if not a or not b:
        return 0.0
    score = max(float(fuzz.token_sort_ratio(a, b)), float(fuzz.token_set_ratio(a, b)) * 0.95)
    if _initials_expand(a.split(), b.split()):
        score = max(score, INITIAL_EXPANSION_SCORE)
    return score


def surnames_agree(a: str, b: str) -> bool:
    """Check that two normalized names share their surnames.

    Parameters
    ----------
    a, b : str
        Normalized names.

    Returns
    -------
    bool
        ``True`` when the paternal surnames are identical and the maternal
        surnames are identical (or one name omits the maternal surname).

    Notes
    -----
    Mexican names end with paternal and maternal surnames. For names with
    three or more tokens the last two tokens are taken as surnames; for a
    two-token name only the last token (paternal surname) is known. This
    guard prevents near-namesakes such as ``"JUAN CARLOS PEREZ LOZANO"``
    (similarity about 89 against ``"JUAN CARLOS PEREZ LOPEZ"``) from being
    revoked silently. Comparisons are exact: an OCR typo in a surname
    yields a human review, never a silent decision.
    """
    tokens_a = a.split()
    tokens_b = b.split()
    if not tokens_a or not tokens_b:
        return False

    def _split(tokens: list[str]) -> tuple[str, Optional[str]]:
        """Return the (paternal, maternal) surnames of a token list.

        Parameters
        ----------
        tokens : list of str
            Name tokens.

        Returns
        -------
        tuple of (str, str or None)
            Paternal surname and maternal surname (``None`` if unknown).
        """
        if len(tokens) >= 3:
            return tokens[-2], tokens[-1]
        return tokens[-1], None

    paternal_a, maternal_a = _split(tokens_a)
    paternal_b, maternal_b = _split(tokens_b)
    if paternal_a != paternal_b:
        return False
    if maternal_a is None or maternal_b is None:
        return True
    return maternal_a == maternal_b


def first_given_name_agrees(a: str, b: str) -> bool:
    """Check that two normalized names share their first given name.

    Parameters
    ----------
    a, b : str
        Normalized names.

    Returns
    -------
    bool
        ``True`` when the first tokens are identical or one is a
        single-letter initial of the other.

    Notes
    -----
    Guards against relatives sharing both surnames, e.g.
    ``"JUANA CARLOTA PEREZ LOPEZ"`` (similarity about 92 against
    ``"JUAN CARLOS PEREZ LOPEZ"``).
    """
    tokens_a = a.split()
    tokens_b = b.split()
    if not tokens_a or not tokens_b:
        return False
    first_a, first_b = tokens_a[0], tokens_b[0]
    if first_a == first_b:
        return True
    if len(first_a) == 1 and first_b.startswith(first_a):
        return True
    return len(first_b) == 1 and first_a.startswith(first_b)


def classify_person_match(a: str, b: str, threshold: float) -> PersonMatch:
    """Decide whether two normalized names denote the same person.

    Parameters
    ----------
    a, b : str
        Normalized names.
    threshold : float
        Revocation similarity threshold (``Settings.fuzzy_match_threshold``).

    Returns
    -------
    PersonMatch
        Score, surname agreement and decision.

    Notes
    -----
    - ``score >= threshold``, surnames agree and first given name agrees
      -> ``MATCH``.
    - ``score >= threshold`` with a surname or given-name conflict -> ``REVIEW``.
    - ``REVIEW_SCORE_FLOOR <= score < threshold`` and surnames agree -> ``REVIEW``.
    - otherwise -> ``NO_MATCH``.
    """
    score = person_similarity(a, b)
    surnames = surnames_agree(a, b)
    given = first_given_name_agrees(a, b)
    if score >= threshold:
        decision = MatchDecision.MATCH if surnames and given else MatchDecision.REVIEW
    elif score >= REVIEW_SCORE_FLOOR and surnames:
        decision = MatchDecision.REVIEW
    else:
        decision = MatchDecision.NO_MATCH
    return PersonMatch(score=score, surnames_agree=surnames, given_name_agrees=given, decision=decision)


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


def _format_amount(amount: float) -> str:
    """Format an amount as ``$1,000,000.00``.

    Parameters
    ----------
    amount : float
        Amount in MXN.

    Returns
    -------
    str
        Currency-formatted amount.
    """
    return f"${amount:,.2f}"


def extract_amount_in_words(source_quote: str, amount: Optional[float] = None) -> Optional[str]:
    """Extract the amount-in-words parenthesis from a capital clause quote.

    Parameters
    ----------
    source_quote : str
        Verbatim clause, for example ``"... $1,000,000.00 (UN MILLON DE PESOS
        00/100 MONEDA NACIONAL)."``.
    amount : float, optional
        Numeric amount; when given, the parenthesis immediately following
        its ``$1,000,000.00`` rendering is preferred.

    Returns
    -------
    str or None
        Stripped parenthesis content, or ``None`` when no parenthesis
        mentioning ``PESO`` is found.
    """
    if amount is not None:
        rendered = _format_amount(amount)
        position = source_quote.find(rendered)
        if position >= 0:
            match = re.match(r"\s*\(([^()]*)\)", source_quote[position + len(rendered) :])
            if match and "PESO" in strip_accents(match.group(1).upper()):
                return match.group(1).strip()
    candidates = [
        group.strip()
        for group in _PARENTHESIS_RE.findall(source_quote)
        if "PESO" in strip_accents(group.upper())
    ]
    return candidates[-1] if candidates else None


def normalize_role(raw_role: str) -> str:
    """Normalize a corporate role or a role-like power grantee for comparison.

    Parameters
    ----------
    raw_role : str
        Role as written (``"Administrador Unico"``) or a power grantee
        written as a role (``"EL ADMINISTRADOR UNICO"``).

    Returns
    -------
    str
        Uppercase, accent-free text without punctuation and without
        leading articles (``"ADMINISTRADOR UNICO"``).
    """
    text = strip_accents(raw_role.upper().strip())
    text = _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", text)).strip()
    return _ROLE_ARTICLE_RE.sub("", text).strip()


def _normalize_quote(text: str) -> str:
    """Normalize a source quote for subject-support checks.

    Parameters
    ----------
    text : str
        Verbatim quote.

    Returns
    -------
    str
        Uppercase, accent-free text with punctuation replaced by spaces
        and whitespace collapsed.
    """
    upper = strip_accents(text.upper())
    return _WHITESPACE_RE.sub(" ", re.sub(r"[^A-Z0-9\s]", " ", upper)).strip()


SUBJECT_SUPPORT_PARTIAL_RATIO: float = 90.0
"""Minimum ``partial_ratio`` of a subject against its quote when the surname tokens are not all present."""


def subject_supported_by_quote(subject: str, source_quote: str) -> bool:
    """Check that a revocation or grant subject is named by the event's own quote.

    Parameters
    ----------
    subject : str
        Raw name of one affected person (one fragment of ``sujeto_afectado``).
    source_quote : str
        ``evidence.source_quote`` of the same event.

    Returns
    -------
    bool
        ``True`` when every surname token of the normalized subject (the
        last two tokens for names with three or more tokens, otherwise the
        last token) occurs as a token of the normalized quote, or when
        ``fuzz.partial_ratio(subject, quote) >=``
        :data:`SUBJECT_SUPPORT_PARTIAL_RATIO`.

    Notes
    -----
    The resolver runs before the grounding check, so this deterministic
    guard prevents a subject that the quote does not mention from
    revoking (or silently keeping) the wrong person.
    """
    subject_norm = normalize_person_name(subject)
    quote_norm = _normalize_quote(source_quote)
    tokens = subject_norm.split()
    if not tokens or not quote_norm:
        return False
    surnames = tokens[-2:] if len(tokens) >= 3 else tokens[-1:]
    quote_tokens = set(quote_norm.split())
    if all(token in quote_tokens for token in surnames):
        return True
    return float(fuzz.partial_ratio(subject_norm, quote_norm)) >= SUBJECT_SUPPORT_PARTIAL_RATIO


def _is_job_title(fragment: str) -> bool:
    """Check whether a subject fragment is only a job title.

    Parameters
    ----------
    fragment : str
        Uppercase, accent-free fragment.

    Returns
    -------
    bool
        ``True`` for fragments such as ``"ADMINISTRADORA UNICA"`` or
        ``"APODERADO GENERAL"``.
    """
    return _JOB_TITLE_RE.match(_WHITESPACE_RE.sub(" ", fragment.strip())) is not None


def split_revocation_subjects(subject: str) -> list[str]:
    """Split a revocation subject naming several people into one name each.

    Parameters
    ----------
    subject : str
        ``sujeto_afectado`` of a ``REVOCATION`` event, for example
        ``"SR. JUAN C. PEREZ LOPEZ Y LIC. MARIA ELENA TORRES VEGA"``.

    Returns
    -------
    list of str
        One raw name per person (``["SR. JUAN C. PEREZ LOPEZ",
        "LIC. MARIA ELENA TORRES VEGA"]``), or ``[subject]`` when no split
        applies.

    Notes
    -----
    Separators are ``","``, ``";"``, the conjunction ``" Y "`` and the
    conjunction ``" E "`` when the next word starts with ``I`` or ``HI``
    (Spanish euphony rule). Splitting is evaluated on the uppercase,
    accent-free text. Additional rules:

    - A subject containing a legal-entity suffix (``S.A.``,
      ``S.A. DE C.V.``, ``S. DE R.L.``, ``S.C.``, ``A.C.``) is split only on
      ``","`` / ``";"`` (entity names often contain ``" Y "``), and a
      fragment that is only a suffix is glued back to the previous one.
    - Trailing job-title fragments (``"..., ADMINISTRADORA UNICA"``) are
      dropped.
    - Two comma-separated fragments where the first has two tokens and the
      second one to three tokens are read as an inverted name
      (``"PEREZ LOPEZ, JUAN CARLOS"`` -> ``"JUAN CARLOS PEREZ LOPEZ"``).
    - If any fragment would have fewer than two name tokens (for example a
      compound surname such as ``"PEREZ Y LOPEZ"``), the whole subject is
      kept as a single name so that the matcher, and not the splitter,
      decides.

    The resolver evaluates the unsplit subject first (see
    :func:`resolve_revocation_subjects`), so a representative whose name
    contains ``" Y "`` is never split apart.
    """
    text = strip_accents(subject.upper().strip())
    has_entity = _ENTITY_SUFFIX_RE.search(text) is not None
    separator = _LIST_SEPARATOR_RE if has_entity else _SUBJECT_SEPARATOR_RE
    fragments: list[str] = []
    for fragment in (part.strip() for part in separator.split(text)):
        if not fragment:
            continue
        if fragments and _ENTITY_SUFFIX_RE.fullmatch(fragment):
            fragments[-1] = f"{fragments[-1]}, {fragment}"
            continue
        fragments.append(fragment)
    dropped_title = False
    if fragments and not _is_job_title(fragments[0]):
        named = [fragment for fragment in fragments if not _is_job_title(fragment)]
        dropped_title = len(named) < len(fragments)
        fragments = named
    if not fragments:
        return [subject]
    if len(fragments) == 2 and not has_entity:
        first_tokens = normalize_person_name(fragments[0]).split()
        second_tokens = normalize_person_name(fragments[1]).split()
        joined_by_comma = re.search(
            rf"{re.escape(fragments[0])}\s*,\s*{re.escape(fragments[1])}", text
        ) is not None
        if joined_by_comma and len(first_tokens) == 2 and 1 <= len(second_tokens) <= 3:
            return [" ".join([*second_tokens, *first_tokens])]
    if len(fragments) < 2:
        return fragments if dropped_title else [subject]
    if any(len(normalize_person_name(fragment).split()) < 2 for fragment in fragments):
        return [subject]
    return fragments


def resolve_revocation_subjects(subject: str, candidate_names: list[str], threshold: float) -> list[str]:
    """Choose how to read a revocation subject against the known representatives.

    Parameters
    ----------
    subject : str
        Raw ``sujeto_afectado``.
    candidate_names : list of str
        Normalized names of the currently active representatives.
    threshold : float
        Revocation similarity threshold.

    Returns
    -------
    list of str
        The person fragments to evaluate.

    Notes
    -----
    Readings are tried in order: the unsplit subject, the output of
    :func:`split_revocation_subjects` and the plain separator split. The
    first reading in which every fragment is a ``MATCH`` for some
    candidate wins; otherwise :func:`split_revocation_subjects` is used
    and unmatched fragments are flagged downstream. This keeps names such
    as ``"JUAN CARLOS PEREZ Y GOMEZ LOPEZ"`` whole when such a
    representative exists.
    """
    default = split_revocation_subjects(subject)
    text = strip_accents(subject.upper().strip())
    plain = [fragment.strip() for fragment in _SUBJECT_SEPARATOR_RE.split(text) if fragment.strip()]
    readings: list[list[str]] = []
    for reading in ([subject], default, plain):
        if reading and reading not in readings:
            readings.append(reading)

    def _all_match(reading: list[str]) -> bool:
        """Return whether every fragment of a reading matches a candidate.

        Parameters
        ----------
        reading : list of str
            Person fragments.

        Returns
        -------
        bool
            ``True`` when each fragment is a ``MATCH`` for some candidate.
        """
        return all(
            any(
                classify_person_match(normalize_person_name(fragment), name, threshold).decision
                is MatchDecision.MATCH
                for name in candidate_names
            )
            for fragment in reading
        )

    for reading in readings:
        if _all_match(reading):
            return reading
    return default


_RevokedPair = tuple[LegalRepresentativeItem, PersonMatch]
"""A revoked representative paired with the match that revoked it."""


class TemporalResolver:
    """Resolve the current corporate state from a multi-act extraction.

    Parameters
    ----------
    settings : Settings
        Provides ``fuzzy_match_threshold``.

    Notes
    -----
    Pure with respect to its input: :meth:`resolve` works on a deep copy.
    """

    def __init__(self, settings: Settings) -> None:
        """Store the settings.

        Parameters
        ----------
        settings : Settings
            Pipeline settings.
        """
        self._settings = settings
        self._threshold = settings.fuzzy_match_threshold

    def resolve(self, raw: RawDeedExtraction, *, job_id: str, doc_sha256: str) -> RawDeedExtraction:
        """Return a new, temporally resolved extraction.

        Parameters
        ----------
        raw : RawDeedExtraction
            Pre-resolution extraction (never mutated).
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.

        Returns
        -------
        RawDeedExtraction
            Deep copy with sorted history, current capital, normalized
            names and revoked representatives / powers removed.
        """
        started = time.perf_counter()
        resolved = raw.model_copy(deep=True)
        resolved.amendment_history = self._sort_history(resolved.amendment_history)
        self._normalize_names(resolved)
        self._resolve_capital(resolved, job_id=job_id, doc_sha256=doc_sha256)
        revoked_names = self._apply_revocations(resolved, job_id=job_id, doc_sha256=doc_sha256)
        reps = resolved.representantes_legales.value or []
        powers = resolved.poderes.value or []
        capital = resolved.capital_social_actual.value
        log_event(
            _LOGGER,
            tag=StatusTag.OK,
            event="temporal_resolution_complete",
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={
                "amendment_events": len(resolved.amendment_history),
                "revoked_representatives": revoked_names,
                "active_representatives": [item.nombre_normalizado for item in reps],
                "active_powers": len(powers),
                "capital_social_actual": capital.monto_numerico if capital else None,
                "capital_escritura_origen": capital.escritura_origen if capital else None,
            },
        )
        return resolved

    @staticmethod
    def _sort_history(events: list[AmendmentEvent]) -> list[AmendmentEvent]:
        """Sort events chronologically with a stable sort.

        Parameters
        ----------
        events : list of AmendmentEvent
            Events in document order.

        Returns
        -------
        list of AmendmentEvent
            Events ordered by ``fecha_acto``; malformed dates sort last.
        """
        return sorted(events, key=lambda event: _parse_iso_date(event.fecha_acto) or date.max)

    @staticmethod
    def _normalize_names(model: RawDeedExtraction) -> None:
        """Fill ``nombre_normalizado`` for every legal representative.

        Parameters
        ----------
        model : RawDeedExtraction
            Model mutated in place.
        """
        for item in model.representantes_legales.value or []:
            item.nombre_normalizado = normalize_person_name(item.nombre)

    def _resolve_capital(self, model: RawDeedExtraction, *, job_id: str, doc_sha256: str) -> None:
        """Set ``capital_social_actual`` to the latest-dated amount.

        Parameters
        ----------
        model : RawDeedExtraction
            Model mutated in place (history already sorted).
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.

        Notes
        -----
        Candidates are the incorporation capital and every
        ``CAPITAL_INCREASE`` event with a ``monto_nuevo``. Ties on the date
        favour the event (it amends the incorporation). When the words
        cannot be parsed from the event quote, the previous words are kept
        and the field is flagged for review.
        """
        field = model.capital_social_actual
        current = field.value
        current_date = _parse_iso_date(current.fecha_acto_vigente) if current else None
        chosen: Optional[AmendmentEvent] = None
        chosen_date: Optional[date] = current_date
        for event in model.amendment_history:
            if event.event_type is not AmendmentType.CAPITAL_INCREASE:
                continue
            if event.monto_nuevo is None:
                field.flag_for_review(
                    f"Capital increase in deed {event.numero_escritura} has no numeric amount"
                )
                continue
            event_date = _parse_iso_date(event.fecha_acto)
            if event_date is None:
                field.flag_for_review(
                    f"Capital increase in deed {event.numero_escritura} has an unparseable date"
                )
                continue
            if chosen_date is None or event_date >= chosen_date:
                chosen = event
                chosen_date = event_date
        if chosen is None or chosen.monto_nuevo is None:
            return
        words = extract_amount_in_words(chosen.evidence.source_quote, chosen.monto_nuevo)
        if words is None:
            words = current.monto_en_letra if current else ""
            field.flag_for_review(
                f"Amount in words not found in capital increase quote of deed {chosen.numero_escritura}"
            )
        previous_amount = current.monto_numerico if current else None
        field.value = CapitalInfo(
            monto_numerico=float(chosen.monto_nuevo),
            moneda=current.moneda if current else "MXN",
            monto_en_letra=words,
            fecha_acto_vigente=chosen.fecha_acto,
            escritura_origen=chosen.numero_escritura,
        )
        field.status = FieldStatus.EXTRACTED
        field.evidence = chosen.evidence.model_copy(deep=True)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="capital_resolved",
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            details={
                "previous_amount": previous_amount,
                "current_amount": field.value.monto_numerico,
                "fecha_acto_vigente": chosen.fecha_acto,
                "escritura_origen": chosen.numero_escritura,
                "page_number": chosen.evidence.page_number,
            },
        )

    def _apply_revocations(self, model: RawDeedExtraction, *, job_id: str, doc_sha256: str) -> list[str]:
        """Remove revoked representatives and their powers.

        Parameters
        ----------
        model : RawDeedExtraction
            Model mutated in place (history already sorted).
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.

        Returns
        -------
        list of str
            Original names of the revoked representatives.

        Notes
        -----
        A ``sujeto_afectado`` naming several people is split with
        :func:`split_revocation_subjects` and every person is evaluated
        separately against the same event. The resolver-owned event fields
        (``matched_canonical_entity``, ``fuzzy_match_score``,
        ``revoked_appointment_evidence``) are reset before resolution.
        """
        reps_field = model.representantes_legales
        powers_field = model.poderes
        reps: list[LegalRepresentativeItem] = list(reps_field.value or [])
        powers: list[PowerOfAttorneyItem] = list(powers_field.value or [])
        revoked_names: list[str] = []
        for event in model.amendment_history:
            event.matched_canonical_entity = None
            event.fuzzy_match_score = None
            event.revoked_appointment_evidence = []
        for event in model.amendment_history:
            if event.event_type is AmendmentType.GRANT_OF_POWERS and event.sujeto_afectado:
                for subject in split_revocation_subjects(event.sujeto_afectado):
                    if not subject_supported_by_quote(subject, event.evidence.source_quote):
                        self._flag_unsupported_subject(
                            event, subject, model, job_id=job_id, doc_sha256=doc_sha256
                        )
                continue
            if event.event_type is not AmendmentType.REVOCATION:
                continue
            if not event.sujeto_afectado:
                reps_field.flag_for_review(
                    f"Revocation in deed {event.numero_escritura} does not name the affected person"
                )
                continue
            event_revoked: list[_RevokedPair] = []
            subjects = resolve_revocation_subjects(
                event.sujeto_afectado, [rep.nombre_normalizado for rep in reps], self._threshold
            )
            for subject in subjects:
                if not subject_supported_by_quote(subject, event.evidence.source_quote):
                    self._flag_unsupported_subject(event, subject, model, job_id=job_id, doc_sha256=doc_sha256)
                    continue
                reps, powers, pairs = self._apply_single_revocation(
                    event, subject, reps, powers, model, job_id=job_id, doc_sha256=doc_sha256
                )
                event_revoked.extend(pairs)
            self._record_revocation_evidence(event, event_revoked)
            revoked_names.extend(rep.nombre for rep, _ in event_revoked)
        self._flag_orphan_powers(reps, powers, model)
        if reps_field.value is not None:
            reps_field.value = reps
        if powers_field.value is not None:
            powers_field.value = powers
        return revoked_names

    @staticmethod
    def _flag_unsupported_subject(
        event: AmendmentEvent, subject: str, model: RawDeedExtraction, *, job_id: str, doc_sha256: str
    ) -> None:
        """Flag an event subject that its own quote does not name.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` or ``GRANT_OF_POWERS`` event.
        subject : str
            Raw name of the unsupported person.
        model : RawDeedExtraction
            Model whose ``representantes_legales`` and ``poderes`` fields
            are flagged for review.
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.

        Notes
        -----
        For a revocation the act is not applied for that subject, so the
        person is neither silently removed nor silently kept.
        """
        if event.event_type is AmendmentType.REVOCATION:
            reason = (
                f"Revocation subject '{subject}' is not supported by the quote of deed "
                f"{event.numero_escritura}; revocation not applied"
            )
            event_name = "revocation_subject_unsupported"
        else:
            reason = (
                f"Grant subject '{subject}' is not supported by the quote of deed {event.numero_escritura}"
            )
            event_name = "grant_subject_unsupported"
        model.representantes_legales.flag_for_review(reason)
        model.poderes.flag_for_review(reason)
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event=event_name,
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            details={
                "subject": subject,
                "deed": event.numero_escritura,
                "event_type": event.event_type.value,
                "page_number": event.evidence.page_number,
            },
        )

    def _flag_orphan_powers(
        self,
        reps: list[LegalRepresentativeItem],
        powers: list[PowerOfAttorneyItem],
        model: RawDeedExtraction,
    ) -> None:
        """Flag active powers whose grantee is no active representative.

        Parameters
        ----------
        reps : list of LegalRepresentativeItem
            Active representatives after resolution.
        powers : list of PowerOfAttorneyItem
            Active powers after resolution.
        model : RawDeedExtraction
            Model whose ``poderes`` field receives review flags.

        Notes
        -----
        A grantee is accounted for when it is a ``MATCH`` for an active
        representative name or equals an active representative role after
        :func:`normalize_role` (``"EL ADMINISTRADOR UNICO"``).
        """
        names = [rep.nombre_normalizado or normalize_person_name(rep.nombre) for rep in reps]
        roles = {normalize_role(rep.cargo) for rep in reps if rep.cargo}
        for power in powers:
            grantee = normalize_person_name(power.otorgado_a)
            if any(
                classify_person_match(grantee, name, self._threshold).decision is MatchDecision.MATCH
                for name in names
            ):
                continue
            if normalize_role(power.otorgado_a) in roles:
                continue
            model.poderes.flag_for_review(
                f"Power grantee '{power.otorgado_a}' does not match any active representative"
            )

    @staticmethod
    def _record_revocation_evidence(event: AmendmentEvent, revoked: list[_RevokedPair]) -> None:
        """Attach the revoked appointments to the revocation event.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event (mutated).
        revoked : list of tuple of (LegalRepresentativeItem, PersonMatch)
            Representatives revoked by the event with their match.

        Notes
        -----
        ``matched_canonical_entity`` lists the original names joined by
        ``"; "``, ``fuzzy_match_score`` keeps the lowest (most conservative)
        score, ``revoked_appointment_evidence`` receives a deep copy of
        each appointment evidence and ``descripcion`` gets a short note
        without the quotes.
        """
        if not revoked:
            return
        event.matched_canonical_entity = "; ".join(rep.nombre for rep, _ in revoked)
        event.fuzzy_match_score = round(min(match.score for _, match in revoked), 2)
        event.revoked_appointment_evidence = [rep.evidence.model_copy(deep=True) for rep, _ in revoked]
        appointment_refs = "; ".join(
            f"{rep.nombre} ({rep.cargo}, appointed {rep.fecha_nombramiento}, page {rep.evidence.page_number})"
            for rep, _ in revoked
        )
        marker = "Revoked appointment evidence:"
        if marker not in event.descripcion:
            event.descripcion = f"{event.descripcion} | {marker} {appointment_refs}"

    def _apply_single_revocation(
        self,
        event: AmendmentEvent,
        subject: str,
        reps: list[LegalRepresentativeItem],
        powers: list[PowerOfAttorneyItem],
        model: RawDeedExtraction,
        *,
        job_id: str,
        doc_sha256: str,
    ) -> tuple[list[LegalRepresentativeItem], list[PowerOfAttorneyItem], list[_RevokedPair]]:
        """Apply the revocation of one person to the active lists.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event.
        subject : str
            Raw name of one revoked person (one fragment of
            ``sujeto_afectado``).
        reps : list of LegalRepresentativeItem
            Currently active representatives.
        powers : list of PowerOfAttorneyItem
            Currently active powers.
        model : RawDeedExtraction
            Model whose list fields receive review flags.
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.

        Returns
        -------
        tuple
            Remaining representatives, remaining powers and the
            ``(representative, match)`` pairs revoked for this person.

        Notes
        -----
        A ``MATCH`` whose appointment is dated after the revocation is
        never revoked. When another appointment of the same person dated
        on or before the revocation is revoked by this event, the later
        entry is a legitimate re-appointment and stays active. When no
        such earlier appointment exists, the revocation predates the only
        appointment (typically a date typo): the person stays listed, but
        ``representantes_legales`` and ``poderes`` are flagged for review
        and ``revocation_date_inconsistent`` is logged as
        ``[REVIEW_REQUIRED]``, so a revoked person is never kept silently.
        """
        subject_norm = normalize_person_name(subject)
        revocation_date = _parse_iso_date(event.fecha_acto)
        kept_reps: list[LegalRepresentativeItem] = []
        revoked: list[_RevokedPair] = []
        later_appointments: list[LegalRepresentativeItem] = []
        any_candidate = False
        for rep in reps:
            match = classify_person_match(subject_norm, rep.nombre_normalizado, self._threshold)
            if match.decision is MatchDecision.NO_MATCH:
                kept_reps.append(rep)
                continue
            any_candidate = True
            if match.decision is MatchDecision.REVIEW:
                kept_reps.append(rep)
                model.representantes_legales.flag_for_review(
                    f"Ambiguous revocation match: '{subject}' vs '{rep.nombre}' "
                    f"(score {match.score:.2f}, surnames agree: {match.surnames_agree}, "
                    f"given name agrees: {match.given_name_agrees})"
                )
                self._log_ambiguous(event, subject, rep.nombre, match, job_id=job_id, doc_sha256=doc_sha256)
                continue
            appointment_date = _parse_iso_date(rep.fecha_nombramiento)
            if revocation_date is not None and appointment_date is not None and appointment_date > revocation_date:
                kept_reps.append(rep)
                later_appointments.append(rep)
                continue
            if revocation_date is None or appointment_date is None:
                model.representantes_legales.flag_for_review(
                    f"Revocation of '{rep.nombre}' applied without a verifiable date order"
                )
            revoked.append((rep, match))

        if later_appointments and not revoked:
            self._flag_date_inconsistency(
                event, subject, later_appointments, model, job_id=job_id, doc_sha256=doc_sha256
            )

        if not revoked:
            if not any_candidate:
                model.representantes_legales.flag_for_review(
                    f"Revocation of '{subject}' in deed {event.numero_escritura} "
                    "did not match any appointed representative"
                )
                log_event(
                    _LOGGER,
                    tag=StatusTag.WARN,
                    event="revocation_unmatched",
                    stage=_STAGE,
                    job_id=job_id,
                    doc_sha256=doc_sha256,
                    details={"subject": subject, "revocation_deed": event.numero_escritura},
                )
            return kept_reps, self._remove_powers(event, subject_norm, [], kept_reps, powers, model), []

        for rep, match in revoked:
            rep.vigente = False
            log_event(
                _LOGGER,
                tag=StatusTag.REVOKED,
                event="representative_revoked",
                stage=_STAGE,
                job_id=job_id,
                doc_sha256=doc_sha256,
                details={
                    "name": rep.nombre,
                    "matched": subject,
                    "score": round(match.score, 2),
                    "revocation_deed": event.numero_escritura,
                    "revocation_date": event.fecha_acto,
                },
            )
        remaining_powers = self._remove_powers(
            event, subject_norm, [rep for rep, _ in revoked], kept_reps, powers, model
        )
        return kept_reps, remaining_powers, revoked

    @staticmethod
    def _flag_date_inconsistency(
        event: AmendmentEvent,
        subject: str,
        later_appointments: list[LegalRepresentativeItem],
        model: RawDeedExtraction,
        *,
        job_id: str,
        doc_sha256: str,
    ) -> None:
        """Flag a matched revocation that predates every appointment of the person.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event.
        subject : str
            Raw name of the revoked person.
        later_appointments : list of LegalRepresentativeItem
            Matched representatives appointed after the revocation date.
        model : RawDeedExtraction
            Model whose ``representantes_legales`` and ``poderes`` fields
            are flagged for review.
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.
        """
        for rep in later_appointments:
            reason = (
                f"Revocation of '{subject}' in deed {event.numero_escritura} dated {event.fecha_acto} "
                f"precedes the appointment of '{rep.nombre}' dated {rep.fecha_nombramiento}; "
                "revocation not applied, verify the act dates"
            )
            model.representantes_legales.flag_for_review(reason)
            model.poderes.flag_for_review(reason)
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="revocation_date_inconsistent",
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            details={
                "subject": subject,
                "revocation_deed": event.numero_escritura,
                "revocation_date": event.fecha_acto,
                "appointments": [
                    {"name": rep.nombre, "fecha_nombramiento": rep.fecha_nombramiento}
                    for rep in later_appointments
                ],
            },
        )

    def _remove_powers(
        self,
        event: AmendmentEvent,
        subject_norm: str,
        revoked_reps: list[LegalRepresentativeItem],
        active_reps: list[LegalRepresentativeItem],
        powers: list[PowerOfAttorneyItem],
        model: RawDeedExtraction,
    ) -> list[PowerOfAttorneyItem]:
        """Remove powers granted to the revoked person.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event.
        subject_norm : str
            Normalized revocation subject.
        revoked_reps : list of LegalRepresentativeItem
            Representatives revoked by this event for this subject.
        active_reps : list of LegalRepresentativeItem
            Representatives that remain active after this subject.
        powers : list of PowerOfAttorneyItem
            Currently active powers.
        model : RawDeedExtraction
            Model whose ``poderes`` field receives review flags.

        Returns
        -------
        list of PowerOfAttorneyItem
            Powers that remain active.

        Notes
        -----
        A power is matched against the revocation subject and the revoked
        canonical names. Power items carry no date, so if the person was
        granted powers again after the revocation (a later
        ``GRANT_OF_POWERS`` event naming them, or a representative
        appointment dated after the revocation) the powers are kept and
        ``poderes`` is flagged for review instead of guessing.

        A power granted to a role (``"EL ADMINISTRADOR UNICO"``) is matched
        against the ``cargo`` of the revoked representatives with
        :func:`normalize_role`. It is removed when no active representative
        holds the same role and the person was not granted powers again;
        otherwise it is kept and ``poderes`` is flagged for review.
        """
        revocation_date = _parse_iso_date(event.fecha_acto)
        regranted = False
        for other in model.amendment_history:
            if other.event_type is not AmendmentType.GRANT_OF_POWERS or not other.sujeto_afectado:
                continue
            other_date = _parse_iso_date(other.fecha_acto)
            if revocation_date is None or other_date is None or other_date <= revocation_date:
                continue
            other_norm = normalize_person_name(other.sujeto_afectado)
            if classify_person_match(subject_norm, other_norm, self._threshold).decision is MatchDecision.MATCH:
                regranted = True
        for rep in model.representantes_legales.value or []:
            appointment_date = _parse_iso_date(rep.fecha_nombramiento)
            if revocation_date is None or appointment_date is None or appointment_date <= revocation_date:
                continue
            rep_norm = rep.nombre_normalizado or normalize_person_name(rep.nombre)
            if classify_person_match(subject_norm, rep_norm, self._threshold).decision is MatchDecision.MATCH:
                regranted = True
        targets = [subject_norm, *(rep.nombre_normalizado for rep in revoked_reps)]
        revoked_roles = {normalize_role(rep.cargo) for rep in revoked_reps if rep.cargo}
        active_roles = {normalize_role(rep.cargo) for rep in active_reps if rep.cargo}
        kept: list[PowerOfAttorneyItem] = []
        for power in powers:
            grantee = normalize_person_name(power.otorgado_a)
            decisions = [classify_person_match(target, grantee, self._threshold) for target in targets]
            if any(match.decision is MatchDecision.MATCH for match in decisions):
                if regranted:
                    kept.append(power)
                    model.poderes.flag_for_review(
                        f"Powers of '{power.otorgado_a}' were revoked in deed {event.numero_escritura} "
                        "and granted again later; verify which grants remain in force"
                    )
                    continue
                power.vigente = False
                continue
            grantee_role = normalize_role(power.otorgado_a)
            if grantee_role and grantee_role in revoked_roles:
                if regranted or grantee_role in active_roles:
                    kept.append(power)
                    model.poderes.flag_for_review(
                        f"Power granted to role '{power.otorgado_a}' was held by a representative revoked "
                        f"in deed {event.numero_escritura} and the role is held or granted again; "
                        "verify which grants remain in force"
                    )
                    continue
                power.vigente = False
                continue
            if any(match.decision is MatchDecision.REVIEW for match in decisions):
                model.poderes.flag_for_review(
                    f"Ambiguous revocation match for power grantee '{power.otorgado_a}'"
                )
            kept.append(power)
        return kept

    @staticmethod
    def _log_ambiguous(
        event: AmendmentEvent, subject: str, candidate: str, match: PersonMatch, *, job_id: str, doc_sha256: str
    ) -> None:
        """Log an ambiguous revocation match requiring human review.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event.
        subject : str
            Raw name of the revoked person being matched.
        candidate : str
            Original name of the candidate representative.
        match : PersonMatch
            Scored comparison.
        job_id : str
            Job identifier for logging.
        doc_sha256 : str
            Document hash for logging.
        """
        log_event(
            _LOGGER,
            tag=StatusTag.REVIEW_REQUIRED,
            event="revocation_match_ambiguous",
            stage=_STAGE,
            job_id=job_id,
            doc_sha256=doc_sha256,
            details={
                "subject": subject,
                "candidate": candidate,
                "score": round(match.score, 2),
                "surnames_agree": match.surnames_agree,
                "given_name_agrees": match.given_name_agrees,
                "revocation_deed": event.numero_escritura,
            },
        )


__all__ = [
    "INITIAL_EXPANSION_SCORE",
    "REVIEW_SCORE_FLOOR",
    "MatchDecision",
    "PersonMatch",
    "TemporalResolver",
    "classify_person_match",
    "first_given_name_agrees",
    "extract_amount_in_words",
    "normalize_person_name",
    "SUBJECT_SUPPORT_PARTIAL_RATIO",
    "normalize_role",
    "person_similarity",
    "resolve_revocation_subjects",
    "split_revocation_subjects",
    "subject_supported_by_quote",
    "strip_accents",
    "surnames_agree",
]
