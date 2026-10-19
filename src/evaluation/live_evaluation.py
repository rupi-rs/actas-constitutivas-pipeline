"""Repeated fully live evaluation of the sample deeds (``actas-cli evaluate-live``).

Each run copies ``data/samples`` and the ground truth into an isolated
temporary project root with an empty cache, processes every sample deed
with the live providers (RapidOCR, ``gpt-6-luna`` extraction and vision
re-read, Decisions predicate), evaluates the outputs against the ground
truth and reads the token usage of every stage from that run's audit
trail. The runs are then aggregated: per-field and per-document score
statistics, critical false positives, evidence hit rate, human-review
routing precision and recall, latency, tokens per stage and the Decisions
cost.

Notes
-----
The real project's ``data/cache`` and ``outputs/`` are never read or
written: every run works under its own temporary root, and an output
directory inside either location is refused. Only the Decisions API has an
assumed price (USD 0.10 per million input tokens); the Responses API
tokens of extraction and vision re-reads are reported as totals with no
price. Standard deviations are sample standard deviations (``0.0`` with a
single value); the median is reported as ``p50``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import platform
import shutil
import statistics
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.config import Settings
from src.core.exceptions import PipelineError, StorageError
from src.core.logger import StatusTag, configure_logging, get_logger, log_event
from src.evaluation.decisions_check import DECISIONS_INPUT_PRICE_PER_MILLION_USD
from src.evaluation.evaluator import EVIDENCE_FIELDS, Evaluator
from src.ingestion.hasher import compute_file_sha256
from src.pipeline.orchestrator import build_orchestrator
from src.providers.llm_provider import SYSTEM_PROMPT_SHA256

_LOGGER = get_logger("evaluation.live")

REPORT_FILENAME: str = "live_evaluation_report.json"
"""JSON report written under the output directory."""

SUMMARY_FILENAME: str = "live_evaluation_summary.md"
"""Spanish Markdown summary written under the output directory."""

STAGE_EVENTS: dict[str, str] = {
    "extraction": "llm_extraction_completed",
    "vlm_reread": "vlm_reread_completed",
    "decisions_predicate": "verifier_completed",
}
"""Audit event that carries the token usage of each live stage."""

RESPONSES_STAGES: tuple[str, ...] = ("extraction", "vlm_reread")
"""Stages served by the Responses API (tokens reported without a price)."""

DECISIONS_STAGE: str = "decisions_predicate"
"""Stage served by the Decisions API (priced per input token)."""

RAPIDOCR_MODULE: str = "rapidocr_onnxruntime"
"""Import name of the required local OCR engine."""


@dataclass
class StageTokens:
    """Token usage of one stage for one document.

    Parameters
    ----------
    calls : int, default 0
        Number of requests that completed.
    input_tokens : int, default 0
        Sum of the reported input tokens.
    output_tokens : int, default 0
        Sum of the reported output tokens.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class DocumentRun:
    """Outcome of one document in one live run.

    Parameters
    ----------
    acta_id : str
        Document identifier.
    final_score : float
        Evaluator final score (``0.0`` when the document failed).
    field_scores : dict of str to float
        Score per contract field.
    field_passed : dict of str to bool
        Pass flag per contract field.
    evidence_hit_rate : float
        Evaluator evidence hit rate.
    critical_false_positives : int
        Revoked representatives listed as active.
    flagged : list of str
        Fields the pipeline routed to human review.
    expected : list of str
        Fields the ground truth expects in human review.
    latency_ms : float or None
        Wall-clock processing time; ``None`` when the document failed.
    tokens : dict of str to StageTokens
        Token usage per live stage.
    review_reasons : dict of str to str
        Review reason per flagged field.
    security_flags : list of str
        Document-level security flags.
    error : str or None, default None
        Failure description when processing failed.
    """

    acta_id: str
    final_score: float
    field_scores: dict[str, float]
    field_passed: dict[str, bool]
    evidence_hit_rate: float
    critical_false_positives: int
    flagged: list[str]
    expected: list[str]
    latency_ms: Optional[float]
    tokens: dict[str, StageTokens] = field(default_factory=dict)
    review_reasons: dict[str, str] = field(default_factory=dict)
    security_flags: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class LiveRun:
    """One complete live run over every sample deed.

    Parameters
    ----------
    run_index : int
        1-based run number.
    started_at : str
        UTC start timestamp (ISO 8601).
    documents : list of DocumentRun
        Per-document outcomes, in ground-truth order.
    ocr_engines : list of str
        Distinct OCR engines reported by the run (proves real OCR was used).
    overall_score : float
        Evaluator overall score of the run.
    """

    run_index: int
    started_at: str
    documents: list[DocumentRun]
    ocr_engines: list[str] = field(default_factory=list)
    overall_score: float = 0.0


def _now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO 8601 timestamp with a ``Z`` suffix.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value: float) -> float:
    """Round a metric for the report.

    Parameters
    ----------
    value : float
        Raw value.

    Returns
    -------
    float
        Value rounded to 4 decimals.
    """
    return round(value, 4)


def describe(values: Sequence[float]) -> dict[str, Optional[float]]:
    """Return mean, minimum, maximum and sample standard deviation.

    Parameters
    ----------
    values : sequence of float
        Observations.

    Returns
    -------
    dict of str to float or None
        ``{"n", "mean", "min", "max", "stdev"}``; statistics are ``None``
        for an empty sequence and ``stdev`` is ``0.0`` for one value.
    """
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "stdev": None}
    return {
        "n": len(values),
        "mean": _round(statistics.fmean(values)),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "stdev": _round(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def describe_latency(values: Sequence[float]) -> dict[str, Optional[float]]:
    """Return mean, median and maximum latency.

    Parameters
    ----------
    values : sequence of float
        Latencies in milliseconds.

    Returns
    -------
    dict of str to float or None
        ``{"n", "mean_ms", "p50_ms", "max_ms"}``; ``None`` when empty.
    """
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "p50_ms": round(statistics.median(values), 1),
        "max_ms": round(max(values), 1),
    }


def parse_stage_tokens(records: Iterable[dict[str, Any]], doc_sha256: str) -> dict[str, StageTokens]:
    """Sum the token usage of each live stage for one document.

    Parameters
    ----------
    records : iterable of dict
        Parsed audit records of one run.
    doc_sha256 : str
        Document hash used to select the records.

    Returns
    -------
    dict of str to StageTokens
        One entry per key of :data:`STAGE_EVENTS`; missing token counts
        (for example Decisions output tokens) add zero.
    """
    totals = {stage: StageTokens() for stage in STAGE_EVENTS}
    events = {event: stage for stage, event in STAGE_EVENTS.items()}
    for record in records:
        stage = events.get(str(record.get("event")))
        if stage is None or record.get("doc_sha256") != doc_sha256:
            continue
        details = record.get("details") or {}
        totals[stage].calls += 1
        input_tokens = details.get("input_tokens")
        output_tokens = details.get("output_tokens")
        if isinstance(input_tokens, int):
            totals[stage].input_tokens += input_tokens
        if isinstance(output_tokens, int):
            totals[stage].output_tokens += output_tokens
    return totals


def decisions_cost_usd(input_tokens: int) -> float:
    """Return the Decisions API cost of a number of input tokens.

    Parameters
    ----------
    input_tokens : int
        Decisions input tokens.

    Returns
    -------
    float
        USD at :data:`DECISIONS_INPUT_PRICE_PER_MILLION_USD` per million.
    """
    return round(input_tokens * DECISIONS_INPUT_PRICE_PER_MILLION_USD / 1_000_000, 8)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """Return a ratio, or ``None`` when the denominator is zero.

    Parameters
    ----------
    numerator : int
        Numerator.
    denominator : int
        Denominator.

    Returns
    -------
    float or None
        Rounded ratio.
    """
    return _round(numerator / denominator) if denominator else None


def _routing(documents: Sequence[DocumentRun]) -> dict[str, Any]:
    """Compare flagged and expected review fields over successful documents.

    Parameters
    ----------
    documents : sequence of DocumentRun
        Documents processed without error.

    Returns
    -------
    dict of str to Any
        True positives, over-review (flagged, not expected), under-review
        (expected, not flagged; the dangerous case), per-field counts,
        precision ``TP / (TP + over)``, recall ``TP / (TP + under)`` and the
        number of documents whose sets match exactly.
    """
    true_positive = over = under = exact = 0
    over_fields: dict[str, int] = {}
    under_fields: dict[str, int] = {}
    flagged_counts: dict[str, int] = {}
    for document in documents:
        flagged, expected = set(document.flagged), set(document.expected)
        true_positive += len(flagged & expected)
        for name in sorted(flagged - expected):
            over += 1
            over_fields[name] = over_fields.get(name, 0) + 1
        for name in sorted(expected - flagged):
            under += 1
            under_fields[name] = under_fields.get(name, 0) + 1
        for name in sorted(flagged):
            flagged_counts[name] = flagged_counts.get(name, 0) + 1
        exact += int(flagged == expected)
    return {
        "documents": len(documents),
        "true_positive": true_positive,
        "over_review": over,
        "under_review": under,
        "over_review_fields": over_fields,
        "under_review_fields": under_fields,
        "flagged_counts": flagged_counts,
        "precision": _ratio(true_positive, true_positive + over),
        "recall": _ratio(true_positive, true_positive + under),
        "exact_match_documents": exact,
    }


def _token_summary(documents: Sequence[DocumentRun]) -> dict[str, Any]:
    """Summarize tokens per stage over documents.

    Parameters
    ----------
    documents : sequence of DocumentRun
        Documents to summarize.

    Returns
    -------
    dict of str to Any
        Per stage: total and per-document mean of calls, input and output
        tokens.
    """
    count = len(documents)
    summary: dict[str, Any] = {}
    for stage in STAGE_EVENTS:
        items = [document.tokens.get(stage, StageTokens()) for document in documents]
        calls = sum(item.calls for item in items)
        input_tokens = sum(item.input_tokens for item in items)
        output_tokens = sum(item.output_tokens for item in items)
        summary[stage] = {
            "total_calls": calls,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "mean_calls_per_document": round(calls / count, 2) if count else None,
            "mean_input_tokens_per_document": round(input_tokens / count, 1) if count else None,
            "mean_output_tokens_per_document": round(output_tokens / count, 1) if count else None,
        }
    return summary


def aggregate_runs(runs: Sequence[LiveRun]) -> dict[str, Any]:
    """Aggregate live runs per document, per field and overall.

    Parameters
    ----------
    runs : sequence of LiveRun
        Completed runs.

    Returns
    -------
    dict of str to Any
        ``{"per_acta": {...}, "overall": {...}}``. Scores include failed
        documents (scored ``0.0`` by the evaluator, listed under
        ``errors``); routing, latency and tokens use successful documents
        only, except tokens which also count requests made before a failure.

    Notes
    -----
    Pure function over the run results, so it is tested offline.
    """
    by_acta: dict[str, list[DocumentRun]] = {}
    for run in runs:
        for document in run.documents:
            by_acta.setdefault(document.acta_id, []).append(document)
    per_acta: dict[str, Any] = {}
    for acta_id, documents in by_acta.items():
        ok = [document for document in documents if document.error is None]
        field_stats: dict[str, Any] = {}
        for name in EVIDENCE_FIELDS:
            scores = [document.field_scores.get(name, 0.0) for document in documents]
            passed = [document.field_passed.get(name, False) for document in documents]
            stats = describe(scores)
            field_stats[name] = {
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
                "pass_rate": _ratio(sum(passed), len(passed)),
            }
        decisions_input = sum(
            document.tokens.get(DECISIONS_STAGE, StageTokens()).input_tokens for document in documents
        )
        per_acta[acta_id] = {
            "runs": len(documents),
            "runs_ok": len(ok),
            "final_score": describe([document.final_score for document in documents]),
            "fields": field_stats,
            "evidence_hit_rate": describe([document.evidence_hit_rate for document in documents]),
            "critical_false_positives": sum(document.critical_false_positives for document in documents),
            "expected_review": sorted(documents[0].expected) if documents else [],
            "routing": _routing(ok),
            "latency": describe_latency([d.latency_ms for d in ok if d.latency_ms is not None]),
            "tokens": _token_summary(documents),
            "decisions_cost_usd": {
                "total": decisions_cost_usd(decisions_input),
                "mean_per_document": (
                    round(decisions_cost_usd(decisions_input) / len(documents), 8)
                    if documents
                    else None
                ),
            },
            "errors": [document.error for document in documents if document.error is not None],
        }
    all_documents = [document for documents in by_acta.values() for document in documents]
    ok_documents = [document for document in all_documents if document.error is None]
    tokens = _token_summary(all_documents)
    decisions_input = tokens[DECISIONS_STAGE]["total_input_tokens"]
    overall = {
        "runs": len(runs),
        "documents": len(all_documents),
        "documents_failed": len(all_documents) - len(ok_documents),
        "final_score": describe([document.final_score for document in all_documents]),
        "overall_score_per_run": describe([run.overall_score for run in runs]),
        "critical_false_positives_total": sum(d.critical_false_positives for d in all_documents),
        "evidence_hit_rate": describe([document.evidence_hit_rate for document in all_documents]),
        "routing": _routing(ok_documents),
        "latency": describe_latency([d.latency_ms for d in ok_documents if d.latency_ms is not None]),
        "tokens": tokens,
        "responses_api_tokens": {
            "input_tokens": sum(tokens[stage]["total_input_tokens"] for stage in RESPONSES_STAGES),
            "output_tokens": sum(tokens[stage]["total_output_tokens"] for stage in RESPONSES_STAGES),
            "price_assumed": None,
        },
        "decisions_api": {
            "input_tokens": decisions_input,
            "price_per_million_input_tokens_usd": DECISIONS_INPUT_PRICE_PER_MILLION_USD,
            "cost_usd": decisions_cost_usd(decisions_input),
        },
    }
    return {"per_acta": per_acta, "overall": overall}


def check_live_prerequisites(settings: Settings) -> None:
    """Fail fast when a fully live evaluation cannot run.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Raises
    ------
    PipelineError
        If ``OPENAI_API_KEY`` is missing or RapidOCR is not installed.
    """
    if not settings.live_mode_available:
        raise PipelineError("evaluate-live requires OPENAI_API_KEY (set it in .env)")
    if importlib.util.find_spec(RAPIDOCR_MODULE) is None:
        raise PipelineError(
            "evaluate-live requires the real RapidOCR engine (rapidocr-onnxruntime); "
            "install it with 'uv sync --extra ocr --locked'"
        )


def _package_version(name: str) -> Optional[str]:
    """Return an installed distribution version.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str or None
        Version, or ``None`` when not installed.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(settings: Settings) -> dict[str, Any]:
    """Describe the evaluation environment.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    dict of str to Any
        Models, SDK and OCR versions, Python and platform, prompt hash and
        the Decisions price assumption.
    """
    return {
        "timestamp": _now(),
        "extraction_model": settings.openai_extraction_model,
        "vision_model": settings.openai_vision_model,
        "decisions_model": settings.openai_decisions_model,
        "openai_base_url_overridden": settings.openai_base_url is not None,
        "openai_sdk_version": _package_version("openai"),
        "rapidocr_onnxruntime_version": _package_version("rapidocr-onnxruntime"),
        "onnxruntime_version": _package_version("onnxruntime"),
        "pymupdf_version": _package_version("pymupdf"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "decisions_price_per_million_input_tokens_usd": DECISIONS_INPUT_PRICE_PER_MILLION_USD,
    }


def _guard_output_dir(settings: Settings, output_dir: Path) -> None:
    """Refuse an output directory inside the real cache or outputs directory.

    Parameters
    ----------
    settings : Settings
        Real project settings.
    output_dir : Path
        Requested output directory.

    Raises
    ------
    PipelineError
        If ``output_dir`` lies inside ``data/cache`` or ``outputs/``.
    """
    resolved = output_dir.resolve()
    for protected in (settings.cache_dir.resolve(), settings.outputs_dir.resolve()):
        if resolved == protected or protected in resolved.parents:
            raise PipelineError(
                f"evaluate-live must not write inside {protected}; choose another --output-dir"
            )


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON Lines audit trail.

    Parameters
    ----------
    path : Path
        Audit log.

    Returns
    -------
    list of dict
        Records; empty when the file does not exist.
    """
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _prepare_root(settings: Settings, root: Path) -> list[Path]:
    """Copy samples and ground truth into an isolated root with an empty cache.

    Parameters
    ----------
    settings : Settings
        Real project settings (source locations).
    root : Path
        Temporary project root.

    Returns
    -------
    list of Path
        Copied sample PDFs, sorted.

    Raises
    ------
    PipelineError
        If no sample or no ground truth is available.
    """
    samples = sorted(settings.samples_dir.glob("acta_*.pdf"))
    if not samples:
        raise PipelineError(
            f"no sample PDFs found in {settings.samples_dir}; run 'actas-cli generate-samples'"
        )
    if not settings.ground_truth_path.is_file():
        raise PipelineError(f"ground truth not found: {settings.ground_truth_path}")
    target_samples = root / "data" / "samples"
    target_samples.mkdir(parents=True)
    (root / "data" / "cache").mkdir(parents=True)
    (root / "data" / "ground_truth").mkdir(parents=True)
    shutil.copy2(settings.ground_truth_path, root / "data" / "ground_truth" / "ground_truth.json")
    copied: list[Path] = []
    for sample in samples:
        destination = target_samples / sample.name
        shutil.copy2(sample, destination)
        copied.append(destination)
    return copied


def run_once(
    settings: Settings, run_index: int, archive_dir: Path, log_level: int = logging.INFO
) -> LiveRun:
    """Process and evaluate every sample deed once, fully live, in isolation.

    Parameters
    ----------
    settings : Settings
        Real project settings (models, key, sample and ground-truth paths).
    run_index : int
        1-based run number.
    archive_dir : Path
        Directory receiving this run's outputs and audit trail.
    log_level : int, default logging.INFO
        Console and audit level used during the run.

    Returns
    -------
    LiveRun
        Per-document outcomes of the run.

    Raises
    ------
    PipelineError
        If the run cannot be prepared or the live providers are unavailable.

    Notes
    -----
    The audit trail is redirected to the temporary root for the duration of
    the run; the caller restores the real one.
    """
    started_at = _now()
    root = Path(tempfile.mkdtemp(prefix=f"actas_live_run{run_index:02d}_"))
    try:
        samples = _prepare_root(settings, root)
        run_settings = replace(
            settings, project_root=root, audit_log_override=root / "logs" / "pipeline_audit.jsonl"
        )
        configure_logging(run_settings.audit_log_path, level=log_level)
        log_event(
            _LOGGER,
            tag=StatusTag.INFO,
            event="live_run_started",
            stage="evaluation",
            details={"run": run_index, "documents": len(samples)},
            message=f"live evaluation run {run_index}: {len(samples)} documents, empty cache",
        )
        orchestrator = build_orchestrator(run_settings, live=True)
        results = {result.acta_id: result for result in orchestrator.process_all(samples)}
        failures = {failure.path.stem: failure for failure in orchestrator.failures}
        report = Evaluator(run_settings).evaluate()
        records = _read_records(run_settings.audit_log_path)
        hashes = {sample.stem: compute_file_sha256(sample) for sample in samples}
        documents: list[DocumentRun] = []
        for evaluation in report.documents:
            acta_id = evaluation.acta_id
            result = results.get(acta_id)
            failure = failures.get(acta_id)
            reasons: dict[str, str] = {}
            error: Optional[str] = None
            if result is not None:
                for name in result.review_fields:
                    reasons[name] = getattr(result.output, name).review_reason or ""
            elif failure is not None:
                error = f"{failure.error_type}: {failure.message}"
            else:
                error = "document not processed"
            documents.append(
                DocumentRun(
                    acta_id=acta_id,
                    final_score=evaluation.final_score,
                    field_scores={score.field: score.score for score in evaluation.field_scores},
                    field_passed={score.field: score.passed for score in evaluation.field_scores},
                    evidence_hit_rate=evaluation.evidence_hit_rate,
                    critical_false_positives=evaluation.critical_false_positives,
                    flagged=list(evaluation.human_review_flagged),
                    expected=list(evaluation.human_review_expected),
                    latency_ms=result.duration_ms if result is not None else None,
                    tokens=parse_stage_tokens(records, hashes.get(acta_id, "")),
                    review_reasons=reasons,
                    security_flags=list(result.security_flags) if result is not None else [],
                    error=error,
                )
            )
        engines = sorted(
            {
                str((record.get("details") or {}).get("engine"))
                for record in records
                if record.get("event") == "ocr_completed"
            }
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        outputs = archive_dir / "outputs"
        outputs.mkdir(exist_ok=True)
        for path in sorted(run_settings.outputs_dir.glob("*.json")):
            shutil.copy2(path, outputs / path.name)
        if run_settings.audit_log_path.is_file():
            shutil.copy2(run_settings.audit_log_path, archive_dir / "pipeline_audit.jsonl")
        return LiveRun(
            run_index=run_index,
            started_at=started_at,
            documents=documents,
            ocr_engines=engines,
            overall_score=report.overall_score,
        )
    except OSError as exc:
        raise StorageError(f"live run {run_index} could not use {root}", {"error": str(exc)}) from exc
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_live_evaluation(
    settings: Settings,
    runs: int,
    output_dir: Path,
    *,
    log_level: int = logging.INFO,
    on_run: Optional[Callable[[LiveRun], None]] = None,
) -> dict[str, Any]:
    """Run the live evaluation ``runs`` times and aggregate the results.

    Parameters
    ----------
    settings : Settings
        Real project settings.
    runs : int
        Number of independent runs (at least 1).
    output_dir : Path
        Directory for the reports and the per-run archives.
    log_level : int, default logging.INFO
        Logging level used during the runs.
    on_run : callable or None, default None
        Called with each completed run (progress reporting).

    Returns
    -------
    dict of str to Any
        Report with ``environment``, ``runs`` and ``aggregates``.

    Raises
    ------
    PipelineError
        If prerequisites are missing, ``runs < 1``, the output directory is
        protected, or a run cannot be prepared.
    """
    if runs < 1:
        raise PipelineError("--runs must be at least 1")
    check_live_prerequisites(settings)
    _guard_output_dir(settings, output_dir)
    env = environment(settings)
    completed: list[LiveRun] = []
    try:
        for index in range(1, runs + 1):
            run = run_once(settings, index, output_dir / "runs" / f"run_{index:02d}", log_level)
            completed.append(run)
            if on_run is not None:
                on_run(run)
    finally:
        configure_logging(settings.audit_log_path, level=log_level)
    return {
        "environment": env,
        "runs_requested": runs,
        "runs": [asdict(run) for run in completed],
        "aggregates": aggregate_runs(completed),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a report value for Markdown.

    Parameters
    ----------
    value : Any
        Number, ``None`` or text.
    digits : int, default 3
        Decimals for floats.

    Returns
    -------
    str
        ``"n/d"`` for ``None``, fixed decimals for floats, ``str`` otherwise.
    """
    if value is None:
        return "n/d"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary_markdown(report: dict[str, Any]) -> str:
    """Render the Spanish Markdown summary of a live evaluation report.

    Parameters
    ----------
    report : dict of str to Any
        Output of :func:`run_live_evaluation`.

    Returns
    -------
    str
        Factual Markdown tables; every number comes from ``report``.
    """
    env = report["environment"]
    aggregates = report["aggregates"]
    overall = aggregates["overall"]
    lines: list[str] = [
        "# Evaluacion en vivo del pipeline",
        "",
        f"Generado: {env['timestamp']}. Corridas: {overall['runs']} "
        f"(solicitadas: {report['runs_requested']}). Documentos evaluados: {overall['documents']} "
        f"(fallidos: {overall['documents_failed']}).",
        "",
        "Cada corrida copia `data/samples` y la verdad de referencia a un directorio temporal "
        "con cache vacio y procesa todo en vivo: RapidOCR local, extraccion y relectura VLM con "
        "la Responses API y verificacion con el predicate de la Decisions API.",
        "",
        "## Entorno",
        "",
        "| Parametro | Valor |",
        "|---|---|",
        f"| Modelo de extraccion | {env['extraction_model']} |",
        f"| Modelo de vision | {env['vision_model']} |",
        f"| Modelo Decisions | {env['decisions_model']} |",
        f"| SDK openai | {_fmt(env['openai_sdk_version'])} |",
        f"| rapidocr-onnxruntime | {_fmt(env['rapidocr_onnxruntime_version'])} |",
        f"| Python | {env['python_version']} |",
        f"| Plataforma | {env['platform']} |",
        "",
        "## Puntaje final por acta",
        "",
        "| Acta | Corridas OK | Media | Min | Max | Desv. estandar | Falsos positivos criticos |",
        "|---|---|---|---|---|---|---|",
    ]
    for acta_id, data in aggregates["per_acta"].items():
        score = data["final_score"]
        lines.append(
            f"| {acta_id} | {data['runs_ok']}/{data['runs']} | {_fmt(score['mean'])} | "
            f"{_fmt(score['min'])} | {_fmt(score['max'])} | {_fmt(score['stdev'])} | "
            f"{data['critical_false_positives']} |"
        )
    final = overall["final_score"]
    per_run = overall["overall_score_per_run"]
    lines += [
        "",
        f"Global por documento: media {_fmt(final['mean'])}, min {_fmt(final['min'])}, "
        f"max {_fmt(final['max'])}, desv. estandar {_fmt(final['stdev'])}. "
        f"Puntaje global por corrida: media {_fmt(per_run['mean'])}, min {_fmt(per_run['min'])}, "
        f"max {_fmt(per_run['max'])}. Falsos positivos criticos totales: "
        f"{overall['critical_false_positives_total']}. Tasa de acierto de evidencia media: "
        f"{_fmt(overall['evidence_hit_rate']['mean'])}.",
        "",
        "## Puntaje por campo",
        "",
    ]
    for acta_id, data in aggregates["per_acta"].items():
        lines += [
            f"### {acta_id}",
            "",
            "| Campo | Media | Min | Max | Tasa de aprobacion |",
            "|---|---|---|---|---|",
        ]
        for name, stats in data["fields"].items():
            lines.append(
                f"| {name} | {_fmt(stats['mean'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} | "
                f"{_fmt(stats['pass_rate'])} |"
            )
        lines.append("")
    lines += [
        "## Enrutamiento a revision humana",
        "",
        "Sobre-revision: campo marcado que no se esperaba. Sub-revision: campo esperado que no se "
        "marco (el caso peligroso). Precision = VP / (VP + sobre-revision); recall = VP / (VP + "
        "sub-revision). Solo documentos procesados sin error.",
        "",
        "| Acta | Esperados | VP | Sobre-revision | Sub-revision | Precision | Recall | "
        "Coincidencia exacta | Campos sobre-revisados | Campos sub-revisados |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for acta_id, data in aggregates["per_acta"].items():
        routing = data["routing"]
        lines.append(
            f"| {acta_id} | {', '.join(data['expected_review']) or 'ninguno'} | "
            f"{routing['true_positive']} | {routing['over_review']} | {routing['under_review']} | "
            f"{_fmt(routing['precision'])} | {_fmt(routing['recall'])} | "
            f"{routing['exact_match_documents']}/{routing['documents']} | "
            f"{_counts(routing['over_review_fields'])} | {_counts(routing['under_review_fields'])} |"
        )
    routing = overall["routing"]
    lines += [
        f"| total | - | {routing['true_positive']} | {routing['over_review']} | "
        f"{routing['under_review']} | {_fmt(routing['precision'])} | {_fmt(routing['recall'])} | "
        f"{routing['exact_match_documents']}/{routing['documents']} | "
        f"{_counts(routing['over_review_fields'])} | {_counts(routing['under_review_fields'])} |",
        "",
        "## Latencia por documento (ms)",
        "",
        "| Acta | Media | p50 | Max |",
        "|---|---|---|---|",
    ]
    for acta_id, data in aggregates["per_acta"].items():
        latency = data["latency"]
        lines.append(
            f"| {acta_id} | {_fmt(latency['mean_ms'], 1)} | {_fmt(latency['p50_ms'], 1)} | "
            f"{_fmt(latency['max_ms'], 1)} |"
        )
    latency = overall["latency"]
    lines += [
        f"| global | {_fmt(latency['mean_ms'], 1)} | {_fmt(latency['p50_ms'], 1)} | "
        f"{_fmt(latency['max_ms'], 1)} |",
        "",
        "## Tokens por etapa (media por documento)",
        "",
        "| Acta | Etapa | Llamadas | Tokens de entrada | Tokens de salida |",
        "|---|---|---|---|---|",
    ]
    for acta_id, data in aggregates["per_acta"].items():
        for stage, stats in data["tokens"].items():
            lines.append(
                f"| {acta_id} | {stage} | {_fmt(stats['mean_calls_per_document'], 2)} | "
                f"{_fmt(stats['mean_input_tokens_per_document'], 1)} | "
                f"{_fmt(stats['mean_output_tokens_per_document'], 1)} |"
            )
    responses = overall["responses_api_tokens"]
    decisions = overall["decisions_api"]
    lines += [
        "",
        "## Costo",
        "",
        f"- Decisions API: {decisions['input_tokens']} tokens de entrada en total; a USD "
        f"{decisions['price_per_million_input_tokens_usd']:.2f} por millon, costo USD "
        f"{decisions['cost_usd']:.6f}.",
        f"- Responses API ({env['extraction_model']}, extraccion y relectura VLM): "
        f"{responses['input_tokens']} tokens de entrada y {responses['output_tokens']} de salida "
        "en total; no se asume precio.",
        "",
        "## Errores",
        "",
    ]
    errors = [
        f"- {acta_id}: {error}"
        for acta_id, data in aggregates["per_acta"].items()
        for error in data["errors"]
    ]
    lines += errors or ["- Ninguno."]
    lines.append("")
    return "\n".join(lines)


def _counts(counts: dict[str, int]) -> str:
    """Format per-field counts for a Markdown cell.

    Parameters
    ----------
    counts : dict of str to int
        Occurrences per field.

    Returns
    -------
    str
        ``"field (n), ..."`` or ``"-"`` when empty.
    """
    return ", ".join(f"{name} ({count})" for name, count in sorted(counts.items())) or "-"


def write_live_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write the JSON report and the Spanish Markdown summary.

    Parameters
    ----------
    report : dict of str to Any
        Output of :func:`run_live_evaluation`.
    output_dir : Path
        Destination directory (created when missing).

    Returns
    -------
    tuple of (Path, Path)
        JSON report and Markdown summary paths.

    Raises
    ------
    StorageError
        If a file cannot be written.
    """
    json_path = output_dir / REPORT_FILENAME
    markdown_path = output_dir / SUMMARY_FILENAME
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_summary_markdown(report), encoding="utf-8")
    except OSError as exc:
        raise StorageError(f"cannot write live evaluation reports to {output_dir}", {"error": str(exc)}) from exc
    return json_path, markdown_path


__all__ = [
    "DocumentRun",
    "LiveRun",
    "StageTokens",
    "aggregate_runs",
    "check_live_prerequisites",
    "decisions_cost_usd",
    "describe",
    "describe_latency",
    "parse_stage_tokens",
    "render_summary_markdown",
    "run_live_evaluation",
    "run_once",
    "write_live_reports",
]
