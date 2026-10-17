"""Logger tests: PII redaction, audit schema and handler hygiene."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.core.logger import (
    PIIRedactionFilter,
    StatusTag,
    configure_logging,
    get_logger,
    log_event,
    redact_text,
    redact_value,
)
from tests.conftest import read_audit_records

REQUIRED_KEYS = {"timestamp", "level", "job_id", "doc_sha256", "stage", "event", "duration_ms", "details"}
PERSON_RFC = "PERJ800101ABC"
COMPANY_RFC = "LTP190510AB1"
CURP = "PERJ800101HJCRPN09"
EMAIL = "carlos.perez@example.com"


def test_person_rfc_is_masked() -> None:
    """A 13-character person RFC keeps its prefix and homoclave only."""
    assert redact_text(f"RFC {PERSON_RFC} registrado") == "RFC PERJ******ABC registrado"


def test_company_rfc_is_masked() -> None:
    """A 12-character company RFC is masked with a six-character date segment."""
    assert redact_text(f"RFC {COMPANY_RFC}") == "RFC LTP******AB1"


def test_curp_is_masked() -> None:
    """A CURP keeps its first four and last two characters."""
    masked = redact_text(f"CURP {CURP}.")
    assert masked == "CURP PERJ************09."
    assert len(masked) == len(f"CURP {CURP}.")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RASC-850412-HM1", "RASC-******-HM1"),
        ("RASC 850412 HM1", "RASC ****** HM1"),
        ("rasc850412hm1", "rasc******hm1"),
        ("rasc-850412-hm1", "rasc-******-hm1"),
        ("LTP-190510-AB1", "LTP-******-AB1"),
    ],
)
def test_rfc_separators_and_case_are_masked(raw: str, expected: str) -> None:
    """RFCs with single separators or lowercase letters are masked in the same shape.

    Parameters
    ----------
    raw : object
        Parametrized input value.
    expected : object
        Parametrized expected value.
    """
    assert redact_text(f"rfc {raw} ok") == f"rfc {expected} ok"


def test_curp_separators_and_case_are_masked() -> None:
    """CURPs with separators or lowercase letters keep prefix and suffix only."""
    assert redact_text("curp RASC-850412-HMCSRR09.") == "curp RASC-******-******09."
    assert redact_text("rasc850412hmcsrr09") == "rasc************09"


def test_email_is_masked() -> None:
    """An email keeps its first local character and domain."""
    assert redact_text(f"contact {EMAIL} now") == "contact c***@example.com now"


def test_redaction_is_idempotent() -> None:
    """Redacting twice yields the same text."""
    once = redact_text(f"{PERSON_RFC} {CURP} {EMAIL}")
    assert redact_text(once) == once
    assert PERSON_RFC not in once and CURP not in once and EMAIL not in once


def test_sha256_hex_is_not_masked() -> None:
    """SHA-256 hex digests are left untouched, in lower and upper case."""
    digest = "822992832a9a659892b2e47ef53f0b8a01e64af1597355bfe86b8aa28f69c697"
    assert redact_text(digest) == digest
    assert redact_text(digest.upper()) == digest.upper()


def test_nested_structures_are_redacted() -> None:
    """Strings nested in dicts, lists and tuples are redacted recursively."""
    value = {"owner": {"rfc": PERSON_RFC, "contacts": [EMAIL, (CURP,)]}, "count": 3, PERSON_RFC: 1}
    redacted = redact_value(value)
    assert redacted == {
        "owner": {"rfc": "PERJ******ABC", "contacts": ["c***@example.com", ("PERJ************09",)]},
        "count": 3,
        "PERJ******ABC": 1,
    }


def test_filter_masks_record_message() -> None:
    """The logging filter redacts interpolated record messages."""
    record = logging.LogRecord("actas.test", logging.INFO, __file__, 1, "rfc %s", (PERSON_RFC,), None)
    assert PIIRedactionFilter().filter(record) is True
    assert record.getMessage() == "rfc PERJ******ABC"


def test_console_output_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    """Console records never contain raw identifiers in message or details.

    Parameters
    ----------
    capsys : object
        Pytest stdout and stderr capture fixture.
    """
    logger = get_logger("pii.console")
    log_event(
        logger,
        tag=StatusTag.INFO,
        event="pii_probe",
        stage="testing",
        details={"rfc": PERSON_RFC, "nested": {"emails": [EMAIL], "curp": CURP}},
        message=f"processing {PERSON_RFC} and {EMAIL} and {CURP}",
    )
    out = capsys.readouterr().out
    assert "[INFO]" in out
    assert "PERJ******ABC" in out
    assert "c***@example.com" in out
    for secret in (PERSON_RFC, EMAIL, CURP, COMPANY_RFC):
        assert secret not in out


def test_audit_output_is_redacted(audit_log_path: Path) -> None:
    """JSONL records never contain raw identifiers in message or details.

    Parameters
    ----------
    audit_log_path : object
        Path of the isolated JSON Lines audit log.
    """
    logger = get_logger("pii.audit")
    log_event(
        logger,
        tag=StatusTag.WARN,
        event="pii_probe",
        stage="testing",
        details={"rfc": PERSON_RFC, "nested": {"emails": [EMAIL], "curp": CURP}},
        message=f"processing {PERSON_RFC} and {EMAIL}",
    )
    raw = audit_log_path.read_text(encoding="utf-8")
    for secret in (PERSON_RFC, EMAIL, CURP):
        assert secret not in raw
    record = read_audit_records(audit_log_path)[-1]
    assert record["details"]["rfc"] == "PERJ******ABC"
    assert record["details"]["nested"]["emails"] == ["c***@example.com"]
    assert record["details"]["nested"]["curp"] == "PERJ************09"
    assert record["message"] == "processing PERJ******ABC and c***@example.com"


def test_audit_records_contain_required_keys(audit_log_path: Path) -> None:
    """Every audit record carries the mandatory contextual fields.

    Parameters
    ----------
    audit_log_path : object
        Path of the isolated JSON Lines audit log.
    """
    logger = get_logger("audit.schema")
    log_event(
        logger,
        tag=StatusTag.REVOKED,
        event="schema_probe",
        stage="testing",
        job_id="job-1",
        doc_sha256="a" * 64,
        duration_ms=12.5,
        details={"k": "v"},
    )
    logger.info("plain message without extras")
    records = read_audit_records(audit_log_path)
    assert len(records) == 2
    for record in records:
        assert REQUIRED_KEYS <= set(record)
    probe = records[0]
    assert probe["job_id"] == "job-1"
    assert probe["doc_sha256"] == "a" * 64
    assert probe["stage"] == "testing"
    assert probe["event"] == "schema_probe"
    assert probe["duration_ms"] == 12.5
    assert probe["details"] == {"k": "v"}
    assert probe["tag"] == "REVOKED"
    assert probe["timestamp"].endswith("Z")
    assert records[1]["job_id"] == "-" and records[1]["details"] == {}


def test_all_status_tags_render_on_console(capsys: pytest.CaptureFixture[str]) -> None:
    """Each allowed tag appears in brackets on the console.

    Parameters
    ----------
    capsys : object
        Pytest stdout and stderr capture fixture.
    """
    logger = get_logger("tags")
    for tag in StatusTag:
        log_event(logger, tag=tag, event="tag_probe", stage="testing")
    out = capsys.readouterr().out
    for tag in StatusTag:
        assert f"[{tag.value}]" in out


def test_invalid_tag_raises_value_error() -> None:
    """An unknown status tag is rejected."""
    with pytest.raises(ValueError):
        log_event(get_logger("invalid"), tag="SUCCESS", event="x", stage="testing")


def test_string_tag_is_accepted(audit_log_path: Path) -> None:
    """A string equal to a StatusTag value is accepted.

    Parameters
    ----------
    audit_log_path : object
        Path of the isolated JSON Lines audit log.
    """
    log_event(get_logger("strtag"), tag="OK", event="string_tag", stage="testing")
    assert read_audit_records(audit_log_path)[-1]["tag"] == "OK"


def test_handlers_are_not_duplicated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Repeated configure and get_logger calls keep a single handler per sink.

    Parameters
    ----------
    tmp_path : object
        Temporary directory provided by pytest.
    capsys : object
        Pytest stdout and stderr capture fixture.
    """
    audit = tmp_path / "dup" / "audit.jsonl"
    for _ in range(3):
        base = configure_logging(audit_log_path=audit)
        get_logger("dup.a")
        get_logger("dup.b")
    pipeline_handlers = [h for h in base.handlers if getattr(h, "_actas_pipeline_handler", False)]
    assert len(pipeline_handlers) == 2
    log_event(get_logger("dup.c"), tag=StatusTag.INFO, event="dup_probe", stage="testing")
    assert capsys.readouterr().out.count("dup_probe") == 1
    assert len(read_audit_records(audit)) == 1


def test_namespace_is_applied_and_propagation_disabled() -> None:
    """Loggers are nested under actas and do not propagate to the root logger."""
    assert get_logger("something").name == "actas.something"
    assert get_logger("actas.other").name == "actas.other"
    assert logging.getLogger("actas").propagate is False
