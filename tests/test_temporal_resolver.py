"""Temporal resolver tests: name matching, capital and revocation resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings
from src.pipeline.resolver import (
    MatchDecision,
    TemporalResolver,
    classify_person_match,
    normalize_person_name,
    person_similarity,
    split_revocation_subjects,
)
from src.pipeline.validators import TEMPORAL_INCONSISTENCY_REASON, verify_temporal_consistency
from src.schemas.deed_models import AmendmentEvent, AmendmentType, RawDeedExtraction
from tests.conftest import read_audit_records


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LIC. JUAN CARLOS PÉREZ LÓPEZ", "JUAN CARLOS PEREZ LOPEZ"),
        ("SR. JUAN C. PEREZ LOPEZ", "JUAN C PEREZ LOPEZ"),
        ("ING. ALEJANDRO GÓMEZ RUIZ", "ALEJANDRO GOMEZ RUIZ"),
        ("C. MARIA ELENA TORRES VEGA", "MARIA ELENA TORRES VEGA"),
        ("  maría   elena  torres vega ", "MARIA ELENA TORRES VEGA"),
        ("MARIA E. TORRES VEGA", "MARIA E TORRES VEGA"),
        ("JUAN C. PEREZ LOPEZ", "JUAN C PEREZ LOPEZ"),
        ("JOSÉ NUÑO ÁLVAREZ", "JOSE NUNO ALVAREZ"),
        ("C.P. JUAN CARLOS PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ"),
        ("CP JUAN CARLOS PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ"),
        ("C. P. JUAN CARLOS PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ"),
        ("CONTADOR PÚBLICO JUAN CARLOS PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ"),
        ("LICDA. MARIA ELENA TORRES VEGA", "MARIA ELENA TORRES VEGA"),
        ("ARQ. ALEJANDRO GÓMEZ RUIZ", "ALEJANDRO GOMEZ RUIZ"),
        ("DON ALEJANDRO GÓMEZ RUIZ", "ALEJANDRO GOMEZ RUIZ"),
        ("DOÑA MARIA ELENA TORRES VEGA", "MARIA ELENA TORRES VEGA"),
        ("EL SEÑOR ALEJANDRO GÓMEZ RUIZ", "ALEJANDRO GOMEZ RUIZ"),
        ("LA SEÑORA MARIA ELENA TORRES VEGA", "MARIA ELENA TORRES VEGA"),
        ("C. PEDRO DIAZ SOTO", "PEDRO DIAZ SOTO"),
        ("DONALDO DIAZ SOTO", "DONALDO DIAZ SOTO"),
    ],
)
def test_normalize_person_name(raw: str, expected: str) -> None:
    """Honorifics and accents are stripped while middle initials are kept."""
    assert normalize_person_name(raw) == expected


def test_normalize_person_name_is_idempotent() -> None:
    """Normalizing an already normalized name changes nothing."""
    once = normalize_person_name("LIC. JUAN CARLOS PÉREZ LÓPEZ")
    assert normalize_person_name(once) == once


def test_juan_carlos_pair_meets_threshold() -> None:
    """The abbreviated revocation name scores about 89.41 against the original."""
    abbreviated = normalize_person_name("SR. JUAN C. PEREZ LOPEZ")
    original = normalize_person_name("LIC. JUAN CARLOS PÉREZ LÓPEZ")
    score = person_similarity(abbreviated, original)
    assert score >= 85.0
    assert score == pytest.approx(89.41, abs=0.05)


def test_unrelated_names_score_below_threshold() -> None:
    """Unrelated people do not reach the fuzzy threshold."""
    score = person_similarity("ALEJANDRO GOMEZ RUIZ", "JUAN CARLOS PEREZ LOPEZ")
    assert score < 85.0
    assert person_similarity("", "JUAN") == 0.0
    decision = classify_person_match("ALEJANDRO GOMEZ RUIZ", "JUAN CARLOS PEREZ LOPEZ", 85.0)
    assert decision.decision is MatchDecision.NO_MATCH


def test_surname_guard_routes_namesake_to_review() -> None:
    """A different maternal surname scores high but is REVIEW, never MATCH."""
    match = classify_person_match("JUAN CARLOS PEREZ LOZANO", "JUAN CARLOS PEREZ LOPEZ", 85.0)
    assert match.score >= 85.0
    assert match.surnames_agree is False
    assert match.decision is MatchDecision.REVIEW


def test_given_name_guard_routes_relative_to_review() -> None:
    """A relative sharing both surnames with a different given name is REVIEW."""
    match = classify_person_match("JUANA CARLOTA PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ", 85.0)
    assert match.decision is MatchDecision.REVIEW


def test_abbreviated_name_is_a_match() -> None:
    """The initial-expanded revocation name is a MATCH."""
    match = classify_person_match("JUAN C PEREZ LOPEZ", "JUAN CARLOS PEREZ LOPEZ", 85.0)
    assert match.decision is MatchDecision.MATCH


def test_acta_c_end_to_end_resolution(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction], audit_log_path: Path
) -> None:
    """Acta C resolves to two active representatives, five powers and the 2024 capital."""
    raw = raw_extractions["acta_c"]
    snapshot = raw.model_dump(mode="json")
    resolved = TemporalResolver(settings).resolve(raw, job_id="job-c", doc_sha256="c" * 64)

    assert raw.model_dump(mode="json") == snapshot
    assert resolved is not raw

    reps = resolved.representantes_legales.value or []
    assert sorted(rep.nombre_normalizado for rep in reps) == [
        "ALEJANDRO GOMEZ RUIZ",
        "MARIA ELENA TORRES VEGA",
    ]
    assert all(rep.vigente for rep in reps)
    assert len(raw.representantes_legales.value or []) == 3

    powers = resolved.poderes.value or []
    assert len(powers) == 5
    grantees = {normalize_person_name(power.otorgado_a) for power in powers}
    assert grantees == {"ALEJANDRO GOMEZ RUIZ", "MARIA ELENA TORRES VEGA"}

    capital = resolved.capital_social_actual.value
    assert capital is not None
    assert capital.monto_numerico == 1000000.0
    assert capital.escritura_origen == "58,400"
    assert capital.fecha_acto_vigente == "2024-08-20"
    assert "UN MILLON" in capital.monto_en_letra
    assert resolved.capital_social_actual.evidence is not None
    assert resolved.capital_social_actual.evidence.page_number == 3
    assert raw.capital_social_actual.value is not None
    assert raw.capital_social_actual.value.monto_numerico == 100000.0

    history = resolved.amendment_history
    assert len(history) == 4
    dates = [event.fecha_acto for event in history]
    assert dates == sorted(dates)
    revocation = next(event for event in history if event.event_type is AmendmentType.REVOCATION)
    assert revocation.matched_canonical_entity == "LIC. JUAN CARLOS PÉREZ LÓPEZ"
    assert revocation.fuzzy_match_score is not None
    assert revocation.fuzzy_match_score >= 85.0
    assert revocation.fuzzy_match_score == pytest.approx(89.41, abs=0.05)
    assert "Revoked appointment evidence" in revocation.descripcion
    juan_raw = next(rep for rep in raw.representantes_legales.value or [] if "JUAN" in rep.nombre)
    assert revocation.revoked_appointment_evidence == [juan_raw.evidence]
    assert revocation.revoked_appointment_evidence[0] is not juan_raw.evidence
    assert juan_raw.evidence.source_quote not in revocation.descripcion
    assert resolved.representantes_legales.requires_human_review is False
    assert resolved.poderes.requires_human_review is False

    records = read_audit_records(audit_log_path)
    revoked = [record for record in records if record["tag"] == "REVOKED"]
    assert len(revoked) == 1
    assert revoked[0]["job_id"] == "job-c"
    assert revoked[0]["details"]["score"] == pytest.approx(89.41, abs=0.05)


@pytest.mark.parametrize("acta_id", ["acta_a", "acta_b"])
def test_actas_without_amendments_are_unchanged(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction], acta_id: str
) -> None:
    """Acta A and B resolve to the same representatives, powers and capital."""
    raw = raw_extractions[acta_id]
    resolved = TemporalResolver(settings).resolve(raw, job_id="job", doc_sha256="d" * 64)
    assert [rep.nombre for rep in resolved.representantes_legales.value or []] == [
        rep.nombre for rep in raw.representantes_legales.value or []
    ]
    assert len(resolved.poderes.value or []) == len(raw.poderes.value or [])
    assert resolved.capital_social_actual.value == raw.capital_social_actual.value
    assert [event.event_type for event in resolved.amendment_history] == [AmendmentType.INCORPORATION]
    assert not any(event.matched_canonical_entity for event in resolved.amendment_history)


def test_appointment_after_revocation_is_kept(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction], audit_log_path: Path
) -> None:
    """A representative appointed after the revocation date is not revoked."""
    raw = raw_extractions["acta_c"]
    juan = next(rep for rep in raw.representantes_legales.value or [] if "JUAN" in rep.nombre)
    juan.fecha_nombramiento = "2025-01-15"
    resolved = TemporalResolver(settings).resolve(raw, job_id="job", doc_sha256="e" * 64)
    names = {rep.nombre_normalizado for rep in resolved.representantes_legales.value or []}
    assert "JUAN CARLOS PEREZ LOPEZ" in names
    revocation = next(
        event for event in resolved.amendment_history if event.event_type is AmendmentType.REVOCATION
    )
    assert revocation.matched_canonical_entity is None
    assert revocation.revoked_appointment_evidence == []
    assert not [record for record in read_audit_records(audit_log_path) if record["tag"] == "REVOKED"]
    assert resolved.representantes_legales.requires_human_review is True
    assert "revocation not applied" in (resolved.representantes_legales.review_reason or "")


def test_namesake_revocation_is_flagged_not_applied(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """A namesake with a different maternal surname triggers review, not removal."""
    raw = raw_extractions["acta_c"]
    revocation = next(
        event for event in raw.amendment_history if event.event_type is AmendmentType.REVOCATION
    )
    revocation.sujeto_afectado = "JUAN CARLOS PEREZ LOZANO"
    resolved = TemporalResolver(settings).resolve(raw, job_id="job", doc_sha256="f" * 64)
    names = {rep.nombre_normalizado for rep in resolved.representantes_legales.value or []}
    assert "JUAN CARLOS PEREZ LOPEZ" in names
    assert resolved.representantes_legales.requires_human_review is True
    assert "Ambiguous revocation match" in (resolved.representantes_legales.review_reason or "")


def _revocation(model: RawDeedExtraction) -> AmendmentEvent:
    """Return the revocation event of an extraction.

    Parameters
    ----------
    model : RawDeedExtraction
        Extraction containing one ``REVOCATION`` event.

    Returns
    -------
    AmendmentEvent
        The revocation event.
    """
    return next(event for event in model.amendment_history if event.event_type is AmendmentType.REVOCATION)


def test_revocation_date_typo_is_flagged_not_silently_kept(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction], audit_log_path: Path
) -> None:
    """A revocation dated 2014 (typo for 2024) keeps Juan Carlos but flags him for review.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    raw_extractions : dict of str to RawDeedExtraction
        Cached raw extractions per acta.
    audit_log_path : Path
        Audit log of the test.
    """
    raw = raw_extractions["acta_c"]
    _revocation(raw).fecha_acto = "2014-08-20"
    resolved = TemporalResolver(settings).resolve(raw, job_id="job-typo", doc_sha256="a" * 64)

    names = {rep.nombre_normalizado for rep in resolved.representantes_legales.value or []}
    assert "JUAN CARLOS PEREZ LOPEZ" in names
    reps_reason = resolved.representantes_legales.review_reason or ""
    assert resolved.representantes_legales.requires_human_review is True
    assert "dated 2014-08-20 precedes the appointment of 'LIC. JUAN CARLOS PÉREZ LÓPEZ'" in reps_reason
    assert resolved.poderes.requires_human_review is True
    assert "revocation not applied" in (resolved.poderes.review_reason or "")

    records = read_audit_records(audit_log_path)
    inconsistent = [record for record in records if record["event"] == "revocation_date_inconsistent"]
    assert len(inconsistent) == 1
    assert inconsistent[0]["tag"] == "REVIEW_REQUIRED"
    assert not [record for record in records if record["tag"] == "REVOKED"]

    issues = verify_temporal_consistency(resolved, job_id="job-typo", doc_sha256="a" * 64)
    assert len(issues) == 1
    assert "revocacion_poderes" in issues[0]
    assert TEMPORAL_INCONSISTENCY_REASON in (resolved.representantes_legales.review_reason or "")


def test_reappointment_after_revocation_is_not_flagged(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """An earlier appointment is revoked while a later re-appointment stays active without a date flag.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    raw_extractions : dict of str to RawDeedExtraction
        Cached raw extractions per acta.
    """
    raw = raw_extractions["acta_c"]
    reps = raw.representantes_legales.value or []
    juan = next(rep for rep in reps if "JUAN" in rep.nombre)
    reappointed = juan.model_copy(deep=True)
    reappointed.fecha_nombramiento = "2025-01-15"
    reps.append(reappointed)
    resolved = TemporalResolver(settings).resolve(raw, job_id="job", doc_sha256="b" * 64)
    active = [rep for rep in resolved.representantes_legales.value or [] if "JUAN" in rep.nombre]
    assert [rep.fecha_nombramiento for rep in active] == ["2025-01-15"]
    assert "revocation not applied" not in (resolved.representantes_legales.review_reason or "")
    assert _revocation(resolved).matched_canonical_entity == juan.nombre


def test_cp_prefixed_revocation_subject_revokes(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """A ``C.P.`` prefixed revocation subject is normalized and revokes the representative.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    raw_extractions : dict of str to RawDeedExtraction
        Cached raw extractions per acta.
    """
    raw = raw_extractions["acta_c"]
    _revocation(raw).sujeto_afectado = "C.P. JUAN CARLOS PEREZ LOPEZ"
    resolved = TemporalResolver(settings).resolve(raw, job_id="job", doc_sha256="c" * 64)
    names = {rep.nombre_normalizado for rep in resolved.representantes_legales.value or []}
    assert "JUAN CARLOS PEREZ LOPEZ" not in names
    assert resolved.representantes_legales.requires_human_review is False


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        (
            "SR. JUAN C. PEREZ LOPEZ Y LIC. MARIA ELENA TORRES VEGA",
            ["SR. JUAN C. PEREZ LOPEZ", "LIC. MARIA ELENA TORRES VEGA"],
        ),
        (
            "JUAN PEREZ LOPEZ, MARIA TORRES VEGA; ANA RUIZ SOTO",
            ["JUAN PEREZ LOPEZ", "MARIA TORRES VEGA", "ANA RUIZ SOTO"],
        ),
        ("JUAN PEREZ LOPEZ E IGNACIO RUIZ SOTO", ["JUAN PEREZ LOPEZ", "IGNACIO RUIZ SOTO"]),
        ("JUAN PEREZ LOPEZ E HILDA RUIZ SOTO", ["JUAN PEREZ LOPEZ", "HILDA RUIZ SOTO"]),
        ("JUAN PEREZ Y LOPEZ", ["JUAN PEREZ Y LOPEZ"]),
        ("ANA E RUIZ SOTO", ["ANA E RUIZ SOTO"]),
        ("SR. JUAN C. PEREZ LOPEZ", ["SR. JUAN C. PEREZ LOPEZ"]),
    ],
)
def test_split_revocation_subjects(subject: str, expected: list[str]) -> None:
    """Multi-person subjects split on separators; compound surnames stay whole.

    Parameters
    ----------
    subject : str
        Raw ``sujeto_afectado``.
    expected : list of str
        Expected person fragments.
    """
    assert split_revocation_subjects(subject) == expected


def test_multi_person_revocation_revokes_both(
    settings: Settings, raw_extractions: dict[str, RawDeedExtraction], audit_log_path: Path
) -> None:
    """A revocation naming two people revokes both and keeps both appointment evidences.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    raw_extractions : dict of str to RawDeedExtraction
        Cached raw extractions per acta.
    audit_log_path : Path
        Audit log of the test.
    """
    raw = raw_extractions["acta_c"]
    _revocation(raw).sujeto_afectado = "SR. JUAN C. PEREZ LOPEZ Y C. MARIA ELENA TORRES VEGA"
    resolved = TemporalResolver(settings).resolve(raw, job_id="job-multi", doc_sha256="d" * 64)

    reps = resolved.representantes_legales.value or []
    assert [rep.nombre_normalizado for rep in reps] == ["ALEJANDRO GOMEZ RUIZ"]
    assert resolved.representantes_legales.requires_human_review is False
    grantees = {normalize_person_name(power.otorgado_a) for power in resolved.poderes.value or []}
    assert grantees == {"ALEJANDRO GOMEZ RUIZ"}

    revocation = _revocation(resolved)
    assert revocation.matched_canonical_entity == "LIC. JUAN CARLOS PÉREZ LÓPEZ; MARIA ELENA TORRES VEGA"
    raw_reps = raw.representantes_legales.value or []
    assert revocation.revoked_appointment_evidence == [raw_reps[0].evidence, raw_reps[1].evidence]
    revoked_records = [r for r in read_audit_records(audit_log_path) if r["tag"] == "REVOKED"]
    assert len(revoked_records) == 2
