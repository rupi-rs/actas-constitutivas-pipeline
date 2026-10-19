"""Offline tests of the ``evaluate-live`` aggregation, reporting and isolation."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pytest
from typer.testing import CliRunner

from src.cli.main import app
from src.core.config import Settings
from src.core.exceptions import PipelineError
from src.evaluation import live_evaluation
from src.evaluation.evaluator import EVIDENCE_FIELDS
from src.evaluation.live_evaluation import (
    DocumentRun,
    LiveRun,
    StageTokens,
    aggregate_runs,
    check_live_prerequisites,
    decisions_cost_usd,
    describe,
    describe_latency,
    parse_stage_tokens,
    render_summary_markdown,
    run_live_evaluation,
    write_live_reports,
)
from src.pipeline.orchestrator import PipelineOrchestrator, build_orchestrator


def _doc(
    acta_id: str,
    final: float,
    *,
    flagged: list[str],
    expected: list[str],
    latency: Optional[float],
    capital_score: float = 1.0,
    decisions_input: int = 0,
    extraction: tuple[int, int] = (0, 0),
    error: Optional[str] = None,
    critical_fp: int = 0,
    hit_rate: float = 1.0,
) -> DocumentRun:
    """Build a fake document outcome.

    Parameters
    ----------
    acta_id : str
        Document identifier.
    final : float
        Final score.
    flagged : list of str
        Fields flagged by the pipeline.
    expected : list of str
        Fields expected in review.
    latency : float or None
        Latency in milliseconds.
    capital_score : float, default 1.0
        Score of ``capital_social_actual`` (other fields score 1.0).
    decisions_input : int, default 0
        Decisions input tokens.
    extraction : tuple of (int, int), default (0, 0)
        Extraction input and output tokens.
    error : str or None, default None
        Failure description.
    critical_fp : int, default 0
        Critical false positives.
    hit_rate : float, default 1.0
        Evidence hit rate.

    Returns
    -------
    DocumentRun
        Fake outcome.
    """
    scores = {name: 1.0 for name in EVIDENCE_FIELDS}
    scores["capital_social_actual"] = capital_score
    return DocumentRun(
        acta_id=acta_id,
        final_score=final,
        field_scores=scores,
        field_passed={name: value >= 1.0 for name, value in scores.items()},
        evidence_hit_rate=hit_rate,
        critical_false_positives=critical_fp,
        flagged=flagged,
        expected=expected,
        latency_ms=latency,
        tokens={
            "extraction": StageTokens(1, extraction[0], extraction[1]),
            "vlm_reread": StageTokens(),
            "decisions_predicate": StageTokens(2 if decisions_input else 0, decisions_input, 0),
        },
        error=error,
    )


def _runs() -> list[LiveRun]:
    """Build two fake runs over two deeds.

    Returns
    -------
    list of LiveRun
        Run 1: both deeds correct, Acta B over-reviews ``socios``. Run 2:
        Acta B misses the capital review (under-review, capital score 0.5)
        and Acta A fails.
    """
    capital = ["capital_social_actual"]
    first = LiveRun(
        run_index=1,
        started_at="t1",
        documents=[
            _doc("acta_a", 1.0, flagged=[], expected=[], latency=1000.0, extraction=(100, 10)),
            _doc(
                "acta_b",
                1.0,
                flagged=["capital_social_actual", "socios"],
                expected=capital,
                latency=3000.0,
                decisions_input=2_000_000,
                extraction=(200, 20),
            ),
        ],
        overall_score=1.0,
    )
    second = LiveRun(
        run_index=2,
        started_at="t2",
        documents=[
            _doc(
                "acta_a", 0.0, flagged=[], expected=[], latency=None, error="ExtractionError: boom",
                hit_rate=0.0,
            ),
            _doc(
                "acta_b",
                0.9,
                flagged=[],
                expected=capital,
                latency=5000.0,
                capital_score=0.5,
                decisions_input=1_000_000,
                extraction=(300, 30),
                critical_fp=1,
            ),
        ],
        overall_score=0.45,
    )
    return [first, second]


def test_describe_and_latency_math() -> None:
    """Mean, extremes, sample standard deviation and median are computed exactly."""
    stats = describe([1.0, 0.9, 0.8])
    assert stats == {"n": 3, "mean": 0.9, "min": 0.8, "max": 1.0, "stdev": 0.1}
    assert describe([0.7])["stdev"] == 0.0
    assert describe([])["mean"] is None
    assert describe_latency([1000.0, 3000.0, 5000.0, 9000.0]) == {
        "n": 4,
        "mean_ms": 4500.0,
        "p50_ms": 4000.0,
        "max_ms": 9000.0,
    }
    assert decisions_cost_usd(1_000_000) == pytest.approx(0.10)


def test_parse_stage_tokens_filters_by_document_and_event() -> None:
    """Token counts are summed per stage for one document only."""
    records: list[dict[str, Any]] = [
        {"event": "llm_extraction_completed", "doc_sha256": "a", "details": {"input_tokens": 10, "output_tokens": 4}},
        {"event": "vlm_reread_completed", "doc_sha256": "a", "details": {"input_tokens": 7, "output_tokens": 2}},
        {"event": "vlm_reread_completed", "doc_sha256": "a", "details": {"input_tokens": 3, "output_tokens": None}},
        {"event": "verifier_completed", "doc_sha256": "a", "details": {"input_tokens": 500}},
        {"event": "verifier_completed", "doc_sha256": "b", "details": {"input_tokens": 999}},
        {"event": "ocr_completed", "doc_sha256": "a", "details": {"input_tokens": 1}},
    ]
    tokens = parse_stage_tokens(records, "a")
    assert tokens["extraction"] == StageTokens(1, 10, 4)
    assert tokens["vlm_reread"] == StageTokens(2, 10, 2)
    assert tokens["decisions_predicate"] == StageTokens(1, 500, 0)


def test_aggregate_runs_per_field_per_acta_and_overall() -> None:
    """Scores, routing, latency, tokens and cost aggregate as documented."""
    aggregates = aggregate_runs(_runs())
    acta_b = aggregates["per_acta"]["acta_b"]
    assert acta_b["final_score"] == {"n": 2, "mean": 0.95, "min": 0.9, "max": 1.0, "stdev": 0.0707}
    capital = acta_b["fields"]["capital_social_actual"]
    assert capital == {"mean": 0.75, "min": 0.5, "max": 1.0, "pass_rate": 0.5}
    assert acta_b["fields"]["socios"]["pass_rate"] == 1.0
    routing = acta_b["routing"]
    assert routing["true_positive"] == 1
    assert routing["over_review"] == 1 and routing["over_review_fields"] == {"socios": 1}
    assert routing["under_review"] == 1 and routing["under_review_fields"] == {"capital_social_actual": 1}
    assert routing["precision"] == 0.5 and routing["recall"] == 0.5
    assert routing["exact_match_documents"] == 0
    assert acta_b["latency"] == {"n": 2, "mean_ms": 4000.0, "p50_ms": 4000.0, "max_ms": 5000.0}
    assert acta_b["critical_false_positives"] == 1
    assert acta_b["decisions_cost_usd"]["total"] == pytest.approx(0.30)
    assert acta_b["decisions_cost_usd"]["mean_per_document"] == pytest.approx(0.15)
    assert acta_b["tokens"]["extraction"]["mean_input_tokens_per_document"] == 250.0
    acta_a = aggregates["per_acta"]["acta_a"]
    assert acta_a["runs_ok"] == 1 and acta_a["errors"] == ["ExtractionError: boom"]
    assert acta_a["final_score"]["mean"] == 0.5
    assert acta_a["routing"]["documents"] == 1
    assert acta_a["latency"]["n"] == 1
    overall = aggregates["overall"]
    assert overall["documents"] == 4 and overall["documents_failed"] == 1
    assert overall["final_score"]["mean"] == pytest.approx(0.725)
    assert overall["overall_score_per_run"]["mean"] == pytest.approx(0.725)
    assert overall["critical_false_positives_total"] == 1
    assert overall["evidence_hit_rate"]["mean"] == 0.75
    assert overall["routing"]["precision"] == 0.5 and overall["routing"]["recall"] == 0.5
    assert overall["latency"]["p50_ms"] == 3000.0
    assert overall["responses_api_tokens"] == {"input_tokens": 600, "output_tokens": 60, "price_assumed": None}
    assert overall["decisions_api"]["input_tokens"] == 3_000_000
    assert overall["decisions_api"]["cost_usd"] == pytest.approx(0.30)


def test_routing_without_flags_reports_undefined_precision() -> None:
    """Precision and recall are ``None`` when their denominators are zero."""
    run = LiveRun(1, "t", [_doc("acta_a", 1.0, flagged=[], expected=[], latency=10.0)], overall_score=1.0)
    routing = aggregate_runs([run])["overall"]["routing"]
    assert routing["precision"] is None and routing["recall"] is None
    assert routing["exact_match_documents"] == 1


def test_reports_are_written_in_spanish_with_report_numbers(tmp_path: Path) -> None:
    """The JSON report and the Spanish summary carry the aggregated numbers.

    Parameters
    ----------
    tmp_path : object
        Temporary directory.
    """
    report = {
        "environment": {
            "timestamp": "2026-01-01T00:00:00Z",
            "extraction_model": "gpt-6-luna",
            "vision_model": "gpt-6-luna",
            "decisions_model": "gpt-6-luna",
            "openai_sdk_version": "3.33.0",
            "rapidocr_onnxruntime_version": "1.4.4",
            "python_version": "3.12.0",
            "platform": "test",
        },
        "runs_requested": 2,
        "runs": [],
        "aggregates": aggregate_runs(_runs()),
    }
    json_path, markdown_path = write_live_reports(report, tmp_path / "live")
    assert json.loads(json_path.read_text(encoding="utf-8"))["runs_requested"] == 2
    summary = markdown_path.read_text(encoding="utf-8")
    assert summary == render_summary_markdown(report)
    assert "# Evaluacion en vivo del pipeline" in summary
    assert "## Enrutamiento a revision humana" in summary
    assert "| acta_b | 2/2 | 0.950 | 0.900 | 1.000 | 0.071 | 1 |" in summary
    assert "capital_social_actual (1)" in summary and "socios (1)" in summary
    assert "costo USD 0.300000" in summary
    assert "- acta_a: ExtractionError: boom" in summary


def test_prerequisites_fail_fast(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key or a missing RapidOCR engine stops the command before any run.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest patcher.
    """
    with pytest.raises(PipelineError, match="OPENAI_API_KEY"):
        check_live_prerequisites(replace(settings, openai_api_key=None))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(PipelineError, match="uv sync --extra ocr --locked"):
        check_live_prerequisites(replace(settings, openai_api_key="sk-test"))


def test_output_dir_inside_real_outputs_or_cache_is_refused(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command never writes inside the project's ``outputs/`` or ``data/cache``.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest patcher.
    """
    monkeypatch.setattr(live_evaluation, "check_live_prerequisites", lambda current: None)
    for target in (settings.outputs_dir / "live", settings.cache_dir):
        with pytest.raises(PipelineError, match="must not write inside"):
            run_live_evaluation(settings, 1, target)
    with pytest.raises(PipelineError, match="at least 1"):
        run_live_evaluation(settings, 0, settings.project_root / "results")


def test_runs_are_isolated_and_archived(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each run uses its own empty-cache root; the real cache and outputs stay untouched.

    The live orchestrator is replaced by the deterministic one fed with a
    copy of the pre-verified caches, so the test is offline.

    Parameters
    ----------
    settings : object
        Isolated workspace settings (plays the real project).
    monkeypatch : object
        Pytest patcher.
    tmp_path : object
        Temporary directory.
    """
    cache_before = {path.name: path.read_bytes() for path in settings.cache_dir.glob("*.json")}
    seen_roots: list[Path] = []

    def fake_build(run_settings: Settings, *, live: bool) -> PipelineOrchestrator:
        """Return a deterministic orchestrator for an isolated run root.

        Parameters
        ----------
        run_settings : Settings
            Settings of the temporary run root.
        live : bool
            Must be ``True``.

        Returns
        -------
        PipelineOrchestrator
            Deterministic orchestrator.
        """
        assert live is True
        assert run_settings.project_root != settings.project_root
        assert list(run_settings.cache_dir.glob("*.json")) == []
        seen_roots.append(run_settings.project_root)
        for path in settings.cache_dir.glob("*.json"):
            shutil.copy2(path, run_settings.cache_dir / path.name)
        return build_orchestrator(run_settings, live=False)

    monkeypatch.setattr(live_evaluation, "check_live_prerequisites", lambda current: None)
    monkeypatch.setattr(live_evaluation, "build_orchestrator", fake_build)
    output_dir = tmp_path / "live"
    report = run_live_evaluation(settings, 2, output_dir)
    assert len(seen_roots) == 2 and seen_roots[0] != seen_roots[1]
    assert all(not root.exists() for root in seen_roots)
    assert not settings.outputs_dir.exists()
    assert {path.name: path.read_bytes() for path in settings.cache_dir.glob("*.json")} == cache_before
    for index in (1, 2):
        archive = output_dir / "runs" / f"run_{index:02d}"
        assert (archive / "outputs" / "acta_b.json").is_file()
        assert (archive / "pipeline_audit.jsonl").is_file()
    overall = report["aggregates"]["overall"]
    assert overall["documents"] == 6 and overall["documents_failed"] == 0
    assert overall["final_score"]["mean"] == 1.0
    assert overall["routing"]["over_review"] == 0 and overall["routing"]["under_review"] == 0
    assert report["aggregates"]["per_acta"]["acta_b"]["routing"]["true_positive"] == 2
    assert report["runs"][0]["ocr_engines"] == ["deterministic-ocr-v1"]
    assert report["environment"]["decisions_model"] == settings.openai_decisions_model


def test_cli_evaluate_live_requires_key(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """``evaluate-live`` without a key exits 1 with an explicit error.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest patcher.
    """
    monkeypatch.setenv("ACTAS_PROJECT_ROOT", str(settings.project_root))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = CliRunner().invoke(app, ["evaluate-live", "--runs", "1"])
    assert result.exit_code == 1
    assert "[ERROR]" in result.output and "OPENAI_API_KEY" in result.output
