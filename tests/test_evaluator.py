"""Evaluator tests: perfect scores, critical false positives and missing outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.config import Settings
from src.evaluation.evaluator import CRITICAL_FP_PENALTY, Evaluator
from src.pipeline.orchestrator import build_orchestrator
from src.schemas.deed_models import LegalRepresentativeItem
from tests.conftest import ACTA_IDS, read_audit_records


def _process_all(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """Run the pipeline over the three samples.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    sample_paths : dict of str to Path
        PDF path per acta identifier.
    """
    orchestrator = build_orchestrator(settings, live=False)
    results = orchestrator.process_all([sample_paths[acta_id] for acta_id in ACTA_IDS])
    assert len(results) == 3 and orchestrator.failures == []


def test_perfect_outputs_score_one(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """Pipeline outputs on the synthetic deeds score 1.0 per acta."""
    _process_all(settings, sample_paths)
    report = Evaluator(settings).evaluate()
    assert [doc.acta_id for doc in report.documents] == list(ACTA_IDS)
    for doc in report.documents:
        assert doc.final_score == 1.0
        assert doc.weighted_score == 1.0
        assert doc.penalty == 0.0
        assert doc.critical_false_positives == 0
        assert doc.evidence_hit_rate == 1.0
        assert doc.human_review_match is True
        assert all(score.passed for score in doc.field_scores)
    assert report.overall_score == 1.0
    assert report.global_penalty == 0.0
    assert report.total_critical_false_positives == 0
    assert (settings.outputs_dir / "evaluation_report.json").is_file()


def test_expected_human_review_is_matched(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """Only Acta B is expected and flagged for capital review."""
    _process_all(settings, sample_paths)
    report = Evaluator(settings).evaluate()
    by_id = {doc.acta_id: doc for doc in report.documents}
    assert by_id["acta_b"].human_review_expected == ["capital_social_actual"]
    assert by_id["acta_b"].human_review_flagged == ["capital_social_actual"]
    assert by_id["acta_a"].human_review_flagged == []
    assert by_id["acta_c"].human_review_flagged == []


def test_revoked_representative_is_critical_false_positive(
    settings: Settings, sample_paths: dict[str, Path], audit_log_path: Path
) -> None:
    """Re-injecting the revoked representative costs 0.25 and logs CRITICAL_RISK."""
    _process_all(settings, sample_paths)
    output_path = settings.outputs_dir / "acta_c.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    template = dict(payload["representantes_legales"]["value"][0])
    template["nombre"] = "LIC. JUAN CARLOS PÉREZ LÓPEZ"
    template["nombre_normalizado"] = "JUAN CARLOS PEREZ LOPEZ"
    LegalRepresentativeItem.model_validate(template)
    payload["representantes_legales"]["value"].append(template)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = Evaluator(settings).evaluate()
    acta_c = next(doc for doc in report.documents if doc.acta_id == "acta_c")
    assert acta_c.critical_false_positives == 1
    assert acta_c.critical_false_positive_names == ["JUAN CARLOS PEREZ LOPEZ"]
    assert acta_c.penalty == 0.25 == CRITICAL_FP_PENALTY
    assert acta_c.final_score == pytest.approx(max(0.0, acta_c.weighted_score - 0.25), abs=1e-3)
    assert report.total_critical_false_positives == 1
    mean_final = sum(doc.final_score for doc in report.documents) / len(report.documents)
    assert report.global_penalty == CRITICAL_FP_PENALTY
    assert report.overall_score == pytest.approx(max(0.0, mean_final - 0.25), abs=1e-3)
    records = [r for r in read_audit_records(audit_log_path) if r["tag"] == "CRITICAL_RISK"]
    assert len(records) == 1
    assert records[0]["event"] == "critical_false_positive"
    assert records[0]["details"]["penalty"] == 0.25


def test_missing_output_scores_zero_without_crashing(
    settings: Settings, sample_paths: dict[str, Path], audit_log_path: Path
) -> None:
    """A missing output file yields a zero-score evaluation for that acta only."""
    _process_all(settings, sample_paths)
    (settings.outputs_dir / "acta_b.json").unlink()
    report = Evaluator(settings).evaluate()
    by_id = {doc.acta_id: doc for doc in report.documents}
    missing = by_id["acta_b"]
    assert missing.final_score == 0.0
    assert missing.weighted_score == 0.0
    assert missing.evidence_hit_rate == 0.0
    assert all(score.score == 0.0 and score.metric == "missing_output" for score in missing.field_scores)
    assert by_id["acta_a"].final_score == 1.0
    assert by_id["acta_c"].final_score == 1.0
    assert report.overall_score == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert any(r["event"] == "output_unavailable" and r["tag"] == "ERROR" for r in read_audit_records(audit_log_path))


def test_invalid_output_scores_zero(settings: Settings, sample_paths: dict[str, Path]) -> None:
    """An output violating the schema is scored 0 instead of crashing."""
    _process_all(settings, sample_paths)
    (settings.outputs_dir / "acta_a.json").write_text('{"razon_social": 1}', encoding="utf-8")
    report = Evaluator(settings).evaluate()
    assert next(doc for doc in report.documents if doc.acta_id == "acta_a").final_score == 0.0


def test_missing_ground_truth_raises(settings: Settings) -> None:
    """Evaluating without ground truth raises EvaluationError."""
    from src.core.exceptions import EvaluationError

    settings.ground_truth_path.unlink()
    with pytest.raises(EvaluationError):
        Evaluator(settings).evaluate()
