"""Structured audit logging with deterministic PII redaction.

Provides dual-output logging for the pipeline:

1. A sober, fixed-width human-readable console stream on ``stdout``.
2. A JSON Lines audit trail (default ``logs/pipeline_audit.jsonl``).

Every record carries ``timestamp``, ``level``, ``tag``, ``job_id``,
``doc_sha256``, ``stage``, ``event``, ``duration_ms`` and ``details``.
A :class:`PIIRedactionFilter` is attached to both handlers so the message,
the exception text and the (recursively walked) ``details`` payload are
masked before either handler emits.

Notes
-----
Redaction rules (applied in this order, case-sensitive, token-bounded):

- CURP (18 chars, ``[A-Z]{4}\\d{6}[HMX][A-Z]{5}[A-Z0-9]\\d``): keep the
  first 4 and last 2 characters, mask the middle 12
  (``RASC850412HMCSRR09`` -> ``RASC************09``). CURPs are masked first
  because a CURP prefix also satisfies the RFC pattern.
- RFC (``[A-ZÑ&]{3,4}\\d{6}[A-Z0-9]{3}``, 12 or 13 chars): keep the
  alphabetic prefix (3 letters for legal entities, 4 for individuals) and
  the 3-character homoclave, mask the 6-digit date segment
  (``PERJ800101ABC`` -> ``PERJ******ABC``, ``ABC800101XY1`` ->
  ``ABC******XY1``). Length is preserved.
- Email: keep the first character of the local part and the full domain
  (``carlos.ramirez@example.com`` -> ``c***@example.com``).

Matches must not be embedded in a longer alphanumeric token, so SHA-256
hex digests and similar identifiers are never altered.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TextIO

LOGGER_NAMESPACE: str = "actas"
"""Root logger name owning the pipeline handlers."""

_HANDLER_MARKER: str = "_actas_pipeline_handler"


class StatusTag(str, Enum):
    """Allowed status tags for console and audit records.

    Notes
    -----
    Rendered on the console as ``[VALUE]``.
    """

    INFO = "INFO"
    OK = "OK"
    WARN = "WARN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REVOKED = "REVOKED"
    ERROR = "ERROR"
    CRITICAL_RISK = "CRITICAL_RISK"


TAG_TO_LEVEL: dict[StatusTag, int] = {
    StatusTag.INFO: logging.INFO,
    StatusTag.OK: logging.INFO,
    StatusTag.WARN: logging.WARNING,
    StatusTag.REVIEW_REQUIRED: logging.WARNING,
    StatusTag.REVOKED: logging.INFO,
    StatusTag.ERROR: logging.ERROR,
    StatusTag.CRITICAL_RISK: logging.ERROR,
}
"""Mapping from status tag to standard logging level."""

_TAG_WIDTH: int = max(len(tag.value) for tag in StatusTag) + 2
_STAGE_WIDTH: int = 12
_EVENT_WIDTH: int = 22

_CURP_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-ZÑ&0-9])([A-Z]{4})(\d{6}[HMX][A-Z]{5}[A-Z0-9])(\d)(?![A-Z0-9])"
)
_RFC_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-ZÑ&0-9])([A-ZÑ&]{3,4})(\d{6})([A-Z0-9]{3})(?![A-Z0-9])"
)
_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})"
)


def _mask_curp(match: re.Match[str]) -> str:
    """Mask a CURP match keeping the first 4 and last 2 characters.

    Parameters
    ----------
    match : re.Match of str
        Match produced by the CURP pattern.

    Returns
    -------
    str
        Masked CURP of the same length.
    """
    full = match.group(0)
    return full[:4] + "*" * (len(full) - 6) + full[-2:]


def _mask_rfc(match: re.Match[str]) -> str:
    """Mask an RFC match keeping the alphabetic prefix and homoclave.

    Parameters
    ----------
    match : re.Match of str
        Match produced by the RFC pattern.

    Returns
    -------
    str
        Masked RFC of the same length with the date segment replaced.
    """
    return match.group(1) + "*" * len(match.group(2)) + match.group(3)


def _mask_email(match: re.Match[str]) -> str:
    """Mask an email address keeping the first local character and domain.

    Parameters
    ----------
    match : re.Match of str
        Match produced by the email pattern.

    Returns
    -------
    str
        Masked email such as ``c***@example.com``.
    """
    return f"{match.group(1)}***@{match.group(2)}"


def redact_text(text: str) -> str:
    """Apply CURP, RFC and email masking to a string.

    Parameters
    ----------
    text : str
        Arbitrary text that may contain personal identifiers.

    Returns
    -------
    str
        Text with all recognised identifiers masked. Idempotent.
    """
    redacted = _CURP_PATTERN.sub(_mask_curp, text)
    redacted = _RFC_PATTERN.sub(_mask_rfc, redacted)
    return _EMAIL_PATTERN.sub(_mask_email, redacted)


def redact_value(value: Any) -> Any:
    """Recursively redact strings inside nested containers.

    Parameters
    ----------
    value : Any
        A string, mapping, list, tuple, set or scalar.

    Returns
    -------
    Any
        A redacted copy. Mappings become ``dict`` (keys are redacted when
        they are strings), sequences keep list/tuple type, sets become
        sorted lists of redacted items, other scalars are returned as-is.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            (redact_text(key) if isinstance(key, str) else key): redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((redact_value(item) for item in value), key=str)
    return value


class PIIRedactionFilter(logging.Filter):
    """Logging filter masking Mexican RFCs, CURPs and emails.

    Notes
    -----
    The filter mutates the record in place: the interpolated message is
    redacted and stored in ``record.msg`` (with ``record.args`` cleared),
    exception text is pre-rendered and redacted into ``record.exc_text``,
    and ``record.details`` is replaced by a recursively redacted copy.
    It never drops records and is safe to apply more than once.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record in place.

        Parameters
        ----------
        record : logging.LogRecord
            Record about to be emitted.

        Returns
        -------
        bool
            Always ``True``.
        """
        record.msg = redact_text(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_text(record.stack_info)
        details = getattr(record, "details", None)
        if details is not None:
            record.details = redact_value(details)
        for attribute in ("job_id", "stage", "event"):
            current = getattr(record, attribute, None)
            if isinstance(current, str):
                setattr(record, attribute, redact_text(current))
        return True


def _level_to_tag(levelno: int) -> StatusTag:
    """Derive a default status tag from a logging level.

    Parameters
    ----------
    levelno : int
        Standard logging level number.

    Returns
    -------
    StatusTag
        ``ERROR`` for ERROR and above, ``WARN`` for WARNING, else ``INFO``.
    """
    if levelno >= logging.ERROR:
        return StatusTag.ERROR
    if levelno >= logging.WARNING:
        return StatusTag.WARN
    return StatusTag.INFO


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract the normalised contextual fields from a log record.

    Parameters
    ----------
    record : logging.LogRecord
        Record possibly enriched through ``extra``.

    Returns
    -------
    dict of str to Any
        Keys ``timestamp``, ``level``, ``tag``, ``job_id``, ``doc_sha256``,
        ``stage``, ``event``, ``duration_ms``, ``details`` and ``message``.

    Notes
    -----
    Records produced by plain ``logger.info(...)`` calls receive defaults:
    tag derived from level, ``-`` identifiers, stage set to the logger
    name suffix, event ``message``, zero duration and empty details.
    """
    raw_tag = getattr(record, "tag", None)
    tag = raw_tag.value if isinstance(raw_tag, StatusTag) else str(raw_tag or _level_to_tag(record.levelno).value)
    details = getattr(record, "details", None)
    if details is None:
        details = {}
    stage_default = record.name.rsplit(".", 1)[-1]
    timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
    return {
        "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": record.levelname,
        "tag": tag,
        "job_id": str(getattr(record, "job_id", "-")),
        "doc_sha256": str(getattr(record, "doc_sha256", "-")),
        "stage": str(getattr(record, "stage", stage_default)),
        "event": str(getattr(record, "event", "message")),
        "duration_ms": float(getattr(record, "duration_ms", 0.0)),
        "details": details,
        "message": record.getMessage(),
    }


class ConsoleFormatter(logging.Formatter):
    """Sober fixed-width console formatter.

    Notes
    -----
    Line layout (no colours, no emojis)::

        2026-10-17T14:03:11Z [OK]              ingestion    hash_computed          job=ab12cd34 doc=3f9a1b2c... 12.4ms {"bytes": 1024}

    ``job`` is truncated to 8 characters and ``doc`` to 8 characters plus
    ``...``. The free-text message is appended in quotes only when it
    differs from the event name. Details are rendered as compact JSON when
    non-empty.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a single console line.

        Parameters
        ----------
        record : logging.LogRecord
            Record to render.

        Returns
        -------
        str
            Formatted line, followed by the exception text if present.
        """
        fields = _record_fields(record)
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tag = f"[{fields['tag']}]"
        doc = str(fields["doc_sha256"])
        doc_short = doc if len(doc) <= 8 else f"{doc[:8]}..."
        job = str(fields["job_id"])[:8]
        parts = [
            timestamp,
            f"{tag:<{_TAG_WIDTH}}",
            f"{fields['stage']:<{_STAGE_WIDTH}}",
            f"{fields['event']:<{_EVENT_WIDTH}}",
            f"job={job}",
            f"doc={doc_short}",
            f"{fields['duration_ms']:.1f}ms",
        ]
        message = str(fields["message"])
        if message and message != fields["event"]:
            parts.append(json.dumps(message, ensure_ascii=False))
        if fields["details"]:
            parts.append(json.dumps(fields["details"], ensure_ascii=False, default=str, sort_keys=True))
        line = " ".join(parts)
        if record.exc_text:
            line = f"{line}\n{record.exc_text}"
        return line


class JsonLinesFormatter(logging.Formatter):
    """Formatter producing one JSON object per line for the audit trail.

    Notes
    -----
    Keys: ``timestamp`` (ISO-8601 UTC, millisecond precision, ``Z`` suffix),
    ``level``, ``tag``, ``logger``, ``job_id``, ``doc_sha256``, ``stage``,
    ``event``, ``duration_ms``, ``details``, ``message`` and, when present,
    ``exception``. Non-ASCII characters are preserved; non-serialisable
    values are converted with ``str``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a JSON line.

        Parameters
        ----------
        record : logging.LogRecord
            Record to render.

        Returns
        -------
        str
            Single-line JSON document.
        """
        fields = _record_fields(record)
        payload: dict[str, Any] = {
            "timestamp": fields["timestamp"],
            "level": fields["level"],
            "tag": fields["tag"],
            "logger": record.name,
            "job_id": fields["job_id"],
            "doc_sha256": fields["doc_sha256"],
            "stage": fields["stage"],
            "event": fields["event"],
            "duration_ms": round(float(fields["duration_ms"]), 3),
            "details": fields["details"],
            "message": fields["message"],
        }
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False, default=str)


class _StdoutHandler(logging.StreamHandler[TextIO]):
    """Stream handler that resolves ``sys.stdout`` at emit time.

    Notes
    -----
    Resolving the stream lazily keeps the handler compatible with output
    capture utilities (for example pytest ``capsys``) that swap
    ``sys.stdout`` after the handler is created.
    """

    def __init__(self) -> None:
        """Initialise the handler bound to the current ``sys.stdout``."""
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to the current ``sys.stdout``.

        Parameters
        ----------
        record : logging.LogRecord
            Record to emit.
        """
        self.stream = sys.stdout
        super().emit(record)


class _LazyJsonLinesFileHandler(logging.FileHandler):
    """Append-mode UTF-8 file handler creating its parent directory lazily.

    Parameters
    ----------
    path : Path
        Audit trail file path.

    Notes
    -----
    The file and its directory are only created when the first record is
    emitted, so importing or configuring logging has no filesystem effect.
    """

    def __init__(self, path: Path) -> None:
        """Initialise the handler without opening the file.

        Parameters
        ----------
        path : Path
            Audit trail file path.
        """
        super().__init__(path, mode="a", encoding="utf-8", delay=True)

    def _open(self) -> io.TextIOWrapper:
        """Create the parent directory and open the file.

        Returns
        -------
        io.TextIOWrapper
            Open text stream in append mode.
        """
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def _remove_pipeline_handlers(base: logging.Logger) -> None:
    """Detach and close handlers previously installed by this module.

    Parameters
    ----------
    base : logging.Logger
        Namespace root logger.
    """
    for handler in list(base.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            base.removeHandler(handler)
            handler.close()


def configure_logging(
    audit_log_path: Optional[Path] = None,
    console: bool = True,
    level: int = logging.INFO,
    audit: bool = True,
) -> logging.Logger:
    """Install (or reinstall) the console and audit handlers.

    Parameters
    ----------
    audit_log_path : Path, optional
        JSON Lines destination. Defaults to ``get_settings().audit_log_path``.
    console : bool, default True
        Whether to attach the stdout console handler.
    level : int, default logging.INFO
        Minimum level for the namespace logger.
    audit : bool, default True
        Whether to attach the JSON Lines audit handler.

    Returns
    -------
    logging.Logger
        The configured namespace root logger (``actas``).

    Notes
    -----
    Safe to call repeatedly: previously installed pipeline handlers are
    closed and replaced, never duplicated. The namespace logger does not
    propagate to the Python root logger.
    """
    from src.core.config import get_settings

    base = logging.getLogger(LOGGER_NAMESPACE)
    _remove_pipeline_handlers(base)
    base.setLevel(level)
    base.propagate = False
    redaction = PIIRedactionFilter()
    if console:
        console_handler = _StdoutHandler()
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.addFilter(redaction)
        setattr(console_handler, _HANDLER_MARKER, True)
        base.addHandler(console_handler)
    if audit:
        target = audit_log_path if audit_log_path is not None else get_settings().audit_log_path
        file_handler = _LazyJsonLinesFileHandler(Path(target))
        file_handler.setFormatter(JsonLinesFormatter())
        file_handler.addFilter(redaction)
        setattr(file_handler, _HANDLER_MARKER, True)
        base.addHandler(file_handler)
    setattr(base, _HANDLER_MARKER, True)
    return base


def reset_logging() -> None:
    """Remove and close all pipeline handlers.

    Notes
    -----
    After a reset the next :func:`get_logger` call reconfigures defaults.
    Intended for tests that need to release the audit file.
    """
    base = logging.getLogger(LOGGER_NAMESPACE)
    _remove_pipeline_handlers(base)
    if getattr(base, _HANDLER_MARKER, False):
        delattr(base, _HANDLER_MARKER)


def get_logger(name: str) -> logging.Logger:
    """Return a pipeline logger, configuring defaults on first use.

    Parameters
    ----------
    name : str
        Component name (for example ``"ingestion"`` or ``__name__``).
        Names outside the ``actas`` namespace are nested under it.

    Returns
    -------
    logging.Logger
        Child of the ``actas`` namespace logger, which owns the handlers.

    Notes
    -----
    Idempotent: handlers live only on the namespace logger and are
    installed once (by :func:`configure_logging`), so repeated calls never
    duplicate output.
    """
    base = logging.getLogger(LOGGER_NAMESPACE)
    if not getattr(base, _HANDLER_MARKER, False):
        configure_logging()
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")


def log_event(
    logger: logging.Logger,
    *,
    tag: StatusTag | str,
    event: str,
    stage: str,
    job_id: str = "-",
    doc_sha256: str = "-",
    duration_ms: float = 0.0,
    details: Optional[Mapping[str, Any]] = None,
    message: Optional[str] = None,
    exc_info: bool = False,
) -> None:
    """Emit a structured pipeline event.

    Parameters
    ----------
    logger : logging.Logger
        Logger obtained from :func:`get_logger`.
    tag : StatusTag or str
        Status tag; strings must equal a :class:`StatusTag` value.
    event : str
        Machine-friendly event name (for example ``"hash_computed"``).
    stage : str
        Pipeline stage (for example ``"ingestion"``, ``"ocr"``).
    job_id : str, default "-"
        Job identifier.
    doc_sha256 : str, default "-"
        Document SHA-256 hex digest.
    duration_ms : float, default 0.0
        Elapsed time for the event in milliseconds.
    details : Mapping of str to Any, optional
        Structured context; redacted recursively before emission.
    message : str, optional
        Free-text message; defaults to ``event``.
    exc_info : bool, default False
        Attach the active exception traceback (redacted).

    Raises
    ------
    ValueError
        If ``tag`` is not a valid :class:`StatusTag` value.

    Notes
    -----
    The logging level is derived from ``TAG_TO_LEVEL``.
    """
    status = tag if isinstance(tag, StatusTag) else StatusTag(tag)
    logger.log(
        TAG_TO_LEVEL[status],
        message if message is not None else event,
        exc_info=exc_info,
        extra={
            "tag": status,
            "event": event,
            "stage": stage,
            "job_id": job_id,
            "doc_sha256": doc_sha256,
            "duration_ms": float(duration_ms),
            "details": dict(details) if details is not None else {},
        },
    )
