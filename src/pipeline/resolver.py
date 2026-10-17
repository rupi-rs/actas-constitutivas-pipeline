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
    "LICENCIADO|LICENCIADA|LIC|INGENIERO|INGENIERA|ING|SENOR|SENORA|SR|SRA|SRITA|"
    "DOCTOR|DOCTORA|DR|DRA|MAESTRO|MAESTRA|MTRO|MTRA|CIUDADANO|CIUDADANA|C"
)
_LEADING_PREFIX_RE: re.Pattern[str] = re.compile(rf"^(?:(?:{_PREFIX_ALTERNATION})\b\.?\s*)+")
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
       stripping would leave nothing, the unstripped text is kept.
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
        """
        reps_field = model.representantes_legales
        powers_field = model.poderes
        reps: list[LegalRepresentativeItem] = list(reps_field.value or [])
        powers: list[PowerOfAttorneyItem] = list(powers_field.value or [])
        revoked_names: list[str] = []
        for event in model.amendment_history:
            if event.event_type is not AmendmentType.REVOCATION:
                continue
            if not event.sujeto_afectado:
                reps_field.flag_for_review(
                    f"Revocation in deed {event.numero_escritura} does not name the affected person"
                )
                continue
            reps, powers, names = self._apply_single_revocation(
                event, reps, powers, model, job_id=job_id, doc_sha256=doc_sha256
            )
            revoked_names.extend(names)
        if reps_field.value is not None:
            reps_field.value = reps
        if powers_field.value is not None:
            powers_field.value = powers
        return revoked_names

    def _apply_single_revocation(
        self,
        event: AmendmentEvent,
        reps: list[LegalRepresentativeItem],
        powers: list[PowerOfAttorneyItem],
        model: RawDeedExtraction,
        *,
        job_id: str,
        doc_sha256: str,
    ) -> tuple[list[LegalRepresentativeItem], list[PowerOfAttorneyItem], list[str]]:
        """Apply one revocation event to the active lists.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event (mutated: canonical match and score).
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
            Remaining representatives, remaining powers and the original
            names of the representatives revoked by this event.
        """
        subject = event.sujeto_afectado or ""
        subject_norm = normalize_person_name(subject)
        revocation_date = _parse_iso_date(event.fecha_acto)
        kept_reps: list[LegalRepresentativeItem] = []
        revoked: list[tuple[LegalRepresentativeItem, PersonMatch]] = []
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
                self._log_ambiguous(event, rep.nombre, match, job_id=job_id, doc_sha256=doc_sha256)
                continue
            appointment_date = _parse_iso_date(rep.fecha_nombramiento)
            if revocation_date is not None and appointment_date is not None and appointment_date > revocation_date:
                kept_reps.append(rep)
                continue
            if revocation_date is None or appointment_date is None:
                model.representantes_legales.flag_for_review(
                    f"Revocation of '{rep.nombre}' applied without a verifiable date order"
                )
            revoked.append((rep, match))

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
            return kept_reps, self._remove_powers(event, subject_norm, [], powers, model), []

        best_rep, best_match = max(revoked, key=lambda pair: pair[1].score)
        event.matched_canonical_entity = best_rep.nombre
        event.fuzzy_match_score = round(best_match.score, 2)
        appointment_refs = "; ".join(
            f"{rep.nombre} ({rep.cargo}, appointed {rep.fecha_nombramiento}, page {rep.evidence.page_number}: "
            f'"{rep.evidence.source_quote}")'
            for rep, _ in revoked
        )
        marker = "Revoked appointment evidence:"
        if marker not in event.descripcion:
            event.descripcion = f"{event.descripcion} | {marker} {appointment_refs}"
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
        revoked_norms = [rep.nombre_normalizado for rep, _ in revoked]
        remaining_powers = self._remove_powers(event, subject_norm, revoked_norms, powers, model)
        return kept_reps, remaining_powers, [rep.nombre for rep, _ in revoked]

    def _remove_powers(
        self,
        event: AmendmentEvent,
        subject_norm: str,
        revoked_norms: list[str],
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
        revoked_norms : list of str
            Normalized names of representatives revoked by this event.
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
        targets = [subject_norm, *revoked_norms]
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
            if any(match.decision is MatchDecision.REVIEW for match in decisions):
                model.poderes.flag_for_review(
                    f"Ambiguous revocation match for power grantee '{power.otorgado_a}'"
                )
            kept.append(power)
        return kept

    @staticmethod
    def _log_ambiguous(
        event: AmendmentEvent, candidate: str, match: PersonMatch, *, job_id: str, doc_sha256: str
    ) -> None:
        """Log an ambiguous revocation match requiring human review.

        Parameters
        ----------
        event : AmendmentEvent
            ``REVOCATION`` event.
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
                "subject": event.sujeto_afectado,
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
    "person_similarity",
    "strip_accents",
    "surnames_agree",
]
