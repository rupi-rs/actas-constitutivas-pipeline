"""Command line presenter for the corporate deed automation pipeline.

Exposes ``generate-samples``, ``process-all``, ``evaluate`` and ``demo``.
This module is the only place where results are rendered for humans; all
runtime events inside the pipeline go through the structured logger.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.core.config import Settings, get_settings
from src.core.exceptions import PipelineError
from src.core.logger import StatusTag, configure_logging, get_logger, log_event
from src.pipeline.orchestrator import ProcessingResult, build_orchestrator
from src.schemas.deed_models import AmendmentType

app = typer.Typer(
    name="actas-cli",
    help="Corporate document automation pipeline for Mexican Actas Constitutivas.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console(highlight=False, width=max(110, shutil.get_terminal_size(fallback=(120, 24)).columns))
_logger = get_logger("cli")

_TAG_STYLES: dict[str, str] = {
    "INFO": "",
    "OK": "green",
    "WARN": "yellow",
    "REVIEW_REQUIRED": "yellow",
    "REVOKED": "cyan",
    "ERROR": "red",
    "CRITICAL_RISK": "bold red",
}
_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def load_dotenv(path: Path) -> int:
    """Load ``KEY=VALUE`` pairs from a dotenv file into the environment.

    Parameters
    ----------
    path : Path
        Dotenv file; a missing file is ignored.

    Returns
    -------
    int
        Number of variables newly set.

    Notes
    -----
    Existing environment variables are never overridden. Blank lines and
    lines starting with ``#`` are skipped; an optional ``export`` prefix and
    matching single or double quotes around values are removed.
    """
    if not path.is_file():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def resolve_log_level(name: Optional[str]) -> int:
    """Translate a ``LOG_LEVEL`` value into a logging level.

    Parameters
    ----------
    name : str or None
        Level name such as ``"DEBUG"`` or ``"INFO"``.

    Returns
    -------
    int
        Matching level; ``logging.INFO`` for empty or unknown names.
    """
    if not name:
        return logging.INFO
    return _LOG_LEVELS.get(name.strip().upper(), logging.INFO)


def _say(tag: StatusTag, message: str) -> None:
    """Print one status line such as ``[OK] message``.

    Parameters
    ----------
    tag : StatusTag
        Status tag.
    message : str
        Message text (printed verbatim, never parsed as markup).
    """
    line = Text(f"[{tag.value}]", style=_TAG_STYLES[tag.value])
    line.append(f" {message}")
    _console.print(line)


def _cell(value: object, style: str = "") -> Text:
    """Return a table cell that is never parsed as markup.

    Parameters
    ----------
    value : object
        Cell value, rendered with ``str``.
    style : str, default ""
        Optional rich style.

    Returns
    -------
    Text
        Plain text cell.
    """
    return Text(str(value), style=style)


def _table(title: str) -> Table:
    """Create a sober table.

    Parameters
    ----------
    title : str
        Table title.

    Returns
    -------
    Table
        Empty table with a simple header style.
    """
    return Table(title=title, box=box.SIMPLE_HEAD, title_justify="left", header_style="bold")


def _settings(ctx: typer.Context) -> Settings:
    """Return the settings stored by the startup callback.

    Parameters
    ----------
    ctx : typer.Context
        Typer context.

    Returns
    -------
    Settings
        Runtime settings.
    """
    settings: Settings = ctx.obj
    return settings


@app.callback()
def startup(ctx: typer.Context) -> None:
    """Load ``.env``, build settings and configure logging once.

    Parameters
    ----------
    ctx : typer.Context
        Typer context; ``ctx.obj`` receives the settings.
    """
    root = Path(os.environ.get("ACTAS_PROJECT_ROOT", "")).resolve() if os.environ.get(
        "ACTAS_PROJECT_ROOT"
    ) else Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    get_settings.cache_clear()
    settings = get_settings()
    configure_logging(settings.audit_log_path, level=resolve_log_level(os.environ.get("LOG_LEVEL")))
    ctx.obj = settings


def _generate_samples(settings: Settings) -> bool:
    """Generate synthetic deeds, caches, ground truth and study guides.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    bool
        ``True`` on success.
    """
    from src.generators.synthetic_deeds import generate_synthetic_deeds

    try:
        deeds = generate_synthetic_deeds(settings)
    except PipelineError as exc:
        _say(StatusTag.ERROR, f"{type(exc).__name__}: {exc.message}")
        return False
    table = _table("Synthetic deeds")
    for column in ("Acta", "SHA-256", "Pages", "PDF"):
        table.add_column(column)
    for deed in deeds:
        table.add_row(
            _cell(deed.acta_id),
            _cell(deed.sha256[:12]),
            _cell(deed.total_pages),
            _cell(deed.pdf_path.relative_to(settings.project_root)),
        )
    _console.print(table)
    _say(StatusTag.OK, f"{len(deeds)} deeds, caches and ground truth written")
    try:
        from src.generators.study.study_pdfs import generate_study_guides
    except ModuleNotFoundError:
        log_event(
            _logger,
            tag=StatusTag.INFO,
            event="study_guides_skipped",
            stage="cli",
            message="study guide generator not present; skipping",
        )
        return True
    try:
        guides = generate_study_guides(settings.study_guides_dir)
    except Exception as exc:
        _say(StatusTag.ERROR, f"study guide generation failed: {type(exc).__name__}: {exc}")
        return False
    _say(StatusTag.OK, f"{len(guides)} study guides written to {settings.study_guides_dir.name}/")
    return True


def _render_results(results: list[ProcessingResult]) -> None:
    """Render the processing summary and per-document notices.

    Parameters
    ----------
    results : list of ProcessingResult
        Successful processing results.
    """
    table = _table("Processing summary")
    for column in (
        "Acta",
        "SHA-256",
        "Pages N/OCR",
        "Cache",
        "Razon social",
        "Capital (MXN)",
        "Active reps",
        "Review fields",
        "Time (ms)",
    ):
        table.add_column(column)
    for result in results:
        output = result.output
        meta = output.audit_metadata
        capital = output.capital_social_actual.value
        reps = output.representantes_legales.value or []
        table.add_row(
            _cell(result.acta_id),
            _cell(result.doc_sha256[:12]),
            _cell(f"{meta.native_pages_count}/{meta.ocr_pages_count}"),
            _cell("hit" if result.cache_hit else "miss"),
            _cell(output.razon_social.value or "-"),
            _cell(f"{capital.monto_numerico:,.2f}" if capital else "-"),
            _cell(sum(1 for rep in reps if rep.vigente)),
            _cell(", ".join(result.review_fields) or "-"),
            _cell(f"{result.duration_ms:.1f}"),
        )
    _console.print(table)
    for result in results:
        output = result.output
        for event in output.amendment_history:
            if event.event_type is not AmendmentType.REVOCATION:
                continue
            score = f"{event.fuzzy_match_score:.2f}" if event.fuzzy_match_score is not None else "n/a"
            _say(
                StatusTag.REVOKED,
                f"{result.acta_id}: {event.sujeto_afectado} matched "
                f"{event.matched_canonical_entity or 'no entity'} (similarity {score}, "
                f"deed {event.numero_escritura}, {event.fecha_acto}); removed from active representatives",
            )
        for name in result.review_fields:
            field = getattr(output, name)
            _say(
                StatusTag.REVIEW_REQUIRED,
                f"{result.acta_id}.{name}: {field.review_reason or 'review required'}",
            )


def _process_all(settings: Settings, live: bool, force_refresh: bool) -> bool:
    """Process every sample deed and render the summary.

    Parameters
    ----------
    settings : Settings
        Runtime settings.
    live : bool
        Whether live providers may be used on cache misses.
    force_refresh : bool
        Whether to bypass the cache.

    Returns
    -------
    bool
        ``True`` if every document was processed.
    """
    paths = sorted(settings.samples_dir.glob("acta_*.pdf"))
    if not paths:
        _say(
            StatusTag.ERROR,
            f"no sample PDFs found in {settings.samples_dir}; run 'actas-cli generate-samples'",
        )
        return False
    try:
        orchestrator = build_orchestrator(settings, live=live)
    except PipelineError as exc:
        _say(StatusTag.ERROR, f"{type(exc).__name__}: {exc.message}")
        return False
    results = orchestrator.process_all(paths, force_refresh=force_refresh)
    for failure in orchestrator.failures:
        _say(StatusTag.ERROR, f"{failure.path.name}: {failure.error_type}: {failure.message}")
    if results:
        _render_results(results)
    failed = len(orchestrator.failures)
    if failed:
        _say(StatusTag.ERROR, f"{failed} of {len(paths)} documents failed")
        return False
    mode = "live" if live and settings.live_mode_available else "deterministic cache"
    _say(StatusTag.OK, f"{len(results)} documents processed ({mode}); outputs in outputs/")
    return True


def _evaluate(settings: Settings) -> bool:
    """Run the evaluator and render its report.

    Parameters
    ----------
    settings : Settings
        Runtime settings.

    Returns
    -------
    bool
        ``True`` if the evaluation completed.
    """
    try:
        from src.evaluation.evaluator import Evaluator

        report = Evaluator(settings).evaluate()
    except (PipelineError, ImportError) as exc:
        message = exc.message if isinstance(exc, PipelineError) else str(exc)
        _say(StatusTag.ERROR, f"evaluation failed: {type(exc).__name__}: {message}")
        return False
    for document in report.documents:
        fields = _table(f"Field metrics: {document.acta_id}")
        for column in ("Field", "Metric", "Score", "Weight", "Weighted", "Pass"):
            fields.add_column(column, justify="left" if column in ("Field", "Metric") else "right")
        for score in document.field_scores:
            fields.add_row(
                _cell(score.field),
                _cell(score.metric),
                _cell(f"{score.score:.3f}"),
                _cell(f"{score.weight:.2f}"),
                _cell(f"{score.weighted:.3f}"),
                _cell("yes" if score.passed else "no", "" if score.passed else "yellow"),
            )
        _console.print(fields)
    summary = _table("Evaluation summary")
    for column in (
        "Acta",
        "Weighted",
        "Penalty",
        "Final",
        "Evidence hit rate",
        "Critical FP",
        "Review routing",
    ):
        summary.add_column(column)
    for document in report.documents:
        routing = "match" if document.human_review_match else "MISMATCH"
        expected = ", ".join(document.human_review_expected) or "none"
        flagged = ", ".join(document.human_review_flagged) or "none"
        summary.add_row(
            _cell(document.acta_id),
            _cell(f"{document.weighted_score:.3f}"),
            _cell(f"{document.penalty:.2f}"),
            _cell(f"{document.final_score:.3f}"),
            _cell(f"{document.evidence_hit_rate:.1%}"),
            _cell(document.critical_false_positives),
            _cell(f"{routing} (expected: {expected}; flagged: {flagged})"),
        )
    _console.print(summary)
    _say(
        StatusTag.INFO,
        f"overall score {report.overall_score:.3f}; overall evidence hit rate "
        f"{report.overall_evidence_hit_rate:.1%}",
    )
    for document in report.documents:
        if document.critical_false_positives > 0:
            names = ", ".join(document.critical_false_positive_names)
            _say(
                StatusTag.CRITICAL_RISK,
                f"{document.acta_id}: revoked representative(s) listed as active: {names}",
            )
    if report.total_critical_false_positives == 0:
        _say(StatusTag.OK, "no critical false positives")
    _say(StatusTag.INFO, f"report written to {settings.outputs_dir.name}/evaluation_report.json")
    return True


@app.command("generate-samples")
def generate_samples(ctx: typer.Context) -> None:
    """Generate sample PDFs, ground truth, caches and study guides.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the settings.

    Raises
    ------
    typer.Exit
        With code 1 on failure.
    """
    if not _generate_samples(_settings(ctx)):
        raise typer.Exit(code=1)


@app.command("process-all")
def process_all(
    ctx: typer.Context,
    live: bool = typer.Option(
        False, "--live/--no-live", help="Use live OpenAI providers on cache misses."
    ),
    force_refresh: bool = typer.Option(
        False, "--force-refresh", help="Ignore existing cache entries."
    ),
) -> None:
    """Process all sample deeds through the pipeline.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the settings.
    live : bool
        Whether live providers may be used on cache misses.
    force_refresh : bool
        Whether to bypass the cache.

    Raises
    ------
    typer.Exit
        With code 1 on failure.
    """
    if not _process_all(_settings(ctx), live, force_refresh):
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(ctx: typer.Context) -> None:
    """Compare outputs with the ground truth and print the report.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the settings.

    Raises
    ------
    typer.Exit
        With code 1 on failure.
    """
    if not _evaluate(_settings(ctx)):
        raise typer.Exit(code=1)


@app.command("demo")
def demo(
    ctx: typer.Context,
    live: bool = typer.Option(
        False, "--live/--no-live", help="Use live OpenAI providers on cache misses."
    ),
) -> None:
    """Run generate-samples, process-all and evaluate end to end.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the settings.
    live : bool
        Whether live providers may be used on cache misses.

    Raises
    ------
    typer.Exit
        With code 1 when any step fails.
    """
    settings = _settings(ctx)
    steps: list[tuple[str, Callable[[], bool]]] = [
        ("generate-samples", lambda: _generate_samples(settings)),
        ("process-all", lambda: _process_all(settings, live, False)),
        ("evaluate", lambda: _evaluate(settings)),
    ]
    for step, action in steps:
        _say(StatusTag.INFO, f"demo step: {step}")
        if not action():
            _say(StatusTag.ERROR, f"demo aborted at step: {step}")
            raise typer.Exit(code=1)
    _say(StatusTag.OK, "demo completed")


if __name__ == "__main__":
    app()
