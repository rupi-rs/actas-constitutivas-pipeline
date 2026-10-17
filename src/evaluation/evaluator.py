"""Evaluation engine comparing pipeline outputs against ground truth.

Implements the weighted metric suite of spec 03 section 3: scalar exact
matches, fuzzy string similarity, entity-list set F1, the critical false
positive penalty for revoked representatives, the evidence hit rate and the
human review routing check.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from rapidfuzz import fuzz

from src.core.config import Settings
from src.core.exceptions import EvaluationError, PipelineError
from src.core.logger import StatusTag, get_logger, log_event
from src.pipeline.resolver import normalize_person_name, person_similarity
from src.providers.interfaces import BaseStorageRepository
from src.schemas.deed_models import CorporateDeedOutput, FieldStatus
from src.storage.repository import FileSystemRepository, save_json

STAGE = "evaluation"
REPORT_FILENAME = "evaluation_report.json"
CRITICAL_FP_PENALTY = 0.25
RAZON_SOCIAL_THRESHOLD = 0.90
PERCENTAGE_TOLERANCE = 0.01
CAPITAL_TOLERANCE = 0.01
FIELD_WEIGHTS: dict[str, float] = {
    "fecha_constitucion": 0.10,
    "numero_escritura": 0.10,
    "capital_social_actual": 0.20,
    "razon_social": 0.10,
    "notario": 0.10,
    "socios": 0.10,
    "representantes_legales": 0.20,
    "poderes": 0.10,
}
EVIDENCE_FIELDS: tuple[str, ...] = tuple(FIELD_WEIGHTS)

_LOGGER = get_logger("evaluator")
_PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


class FieldScore(BaseModel):
    """Score of one evaluated field.

    Parameters
    ----------
    field : str
        Field name.
    metric : str
        Metric identifier.
    score : float
        Raw score in ``[0, 1]``.
    weight : float
        Field weight.
    weighted : float
        ``score * weight``.
    passed : bool
        Whether the field is considered correct.
    detail : str
        Human readable explanation.
    """

    field: str
    metric: str
    score: float
    weight: float
    weighted: float
    passed: bool
    detail: str


class DocumentEvaluation(BaseModel):
    """Evaluation result of one document.

    Parameters
    ----------
    acta_id : str
        Document identifier.
    field_scores : list of FieldScore
        Per-field scores.
    weighted_score : float
        Sum of weighted field scores.
    penalty : float
        Critical false positive penalty applied.
    final_score : float
        ``max(0, weighted_score - penalty)``.
    evidence_hit_rate : float
        Fraction of extracted fields with correct page and verified grounding.
    critical_false_positives : int
        Number of revoked representatives still active.
    critical_false_positive_names : list of str
        Names of those representatives.
    human_review_expected : list of str
        Fields expected to require human review.
    human_review_flagged : list of str
        Fields flagged for human review by the pipeline.
    human_review_match : bool
        Whether both sets are equal.
    """

    acta_id: str
    field_scores: list[FieldScore]
    weighted_score: float
    penalty: float
    final_score: float
    evidence_hit_rate: float
    critical_false_positives: int
    critical_false_positive_names: list[str] = Field(default_factory=list)
    human_review_expected: list[str] = Field(default_factory=list)
    human_review_flagged: list[str] = Field(default_factory=list)
    human_review_match: bool = True


class EvaluationReport(BaseModel):
    """Aggregate evaluation report.

    Parameters
    ----------
    documents : list of DocumentEvaluation
        Per-document results.
    overall_score : float
        Mean final score minus ``global_penalty`` (floor 0).
    overall_evidence_hit_rate : float
        Mean evidence hit rate.
    total_critical_false_positives : int
        Sum of critical false positives.
    generated_at : str
        ISO 8601 UTC timestamp.
    global_penalty : float, default 0.0
        Penalty applied once to ``overall_score`` when any critical false
        positive exists (spec 03: -0.25 to the global score).
    """

    documents: list[DocumentEvaluation]
    overall_score: float
    overall_evidence_hit_rate: float
    total_critical_false_positives: int
    generated_at: str
    global_penalty: float = 0.0


def _r(value: float) -> float:
    """Round a float to four decimals.

    Parameters
    ----------
    value : float
        Value to round.

    Returns
    -------
    float
        Rounded value.
    """
    return round(value, 4)


def _normalize_text(text: str) -> str:
    """Normalize free text for fuzzy comparison.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        Uppercase, accent-free text with punctuation turned into spaces.

    Notes
    -----
    Accents are stripped through NFKD decomposition and runs of whitespace
    are collapsed.
    """
    decomposed = unicodedata.normalize("NFKD", text.upper())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", plain)).strip()


def _digits(text: str) -> str:
    """Keep only the digits of a string.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Digits only.
    """
    return "".join(ch for ch in text if ch.isdigit())


def _make_score(field: str, metric: str, score: float, passed: bool, detail: str) -> FieldScore:
    """Build a :class:`FieldScore` using the configured field weight.

    Parameters
    ----------
    field : str
        Field name.
    metric : str
        Metric identifier.
    score : float
        Raw score.
    passed : bool
        Pass flag.
    detail : str
        Explanation.

    Returns
    -------
    FieldScore
        Rounded score record.
    """
    weight = FIELD_WEIGHTS[field]
    clipped = min(1.0, max(0.0, score))
    return FieldScore(
        field=field,
        metric=metric,
        score=_r(clipped),
        weight=weight,
        weighted=_r(clipped * weight),
        passed=passed,
        detail=detail,
    )


def match_names(
    predicted: list[str], expected: list[str], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedily pair predicted and expected names one-to-one.

    Parameters
    ----------
    predicted : list of str
        Predicted names.
    expected : list of str
        Ground truth names.
    threshold : float
        Minimum similarity (0-100) of a pair.

    Returns
    -------
    list of tuple
        ``(predicted_index, expected_index, score)`` pairs, best first.

    Notes
    -----
    Similarity is :func:`person_similarity` over names normalized with
    :func:`normalize_person_name`; pairs are taken in descending score order
    so each name is used at most once.
    """
    candidates: list[tuple[float, int, int]] = []
    pred_norm = [normalize_person_name(name) for name in predicted]
    exp_norm = [normalize_person_name(name) for name in expected]
    for i, p_name in enumerate(pred_norm):
        for j, e_name in enumerate(exp_norm):
            score = person_similarity(p_name, e_name)
            if score >= threshold:
                candidates.append((score, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_p: set[int] = set()
    used_e: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in used_p or j in used_e:
            continue
        used_p.add(i)
        used_e.add(j)
        pairs.append((i, j, score))
    return pairs


def _prf(matched: int, n_pred: int, n_exp: int) -> tuple[float, float, float]:
    """Compute precision, recall and F1.

    Parameters
    ----------
    matched : int
        True positives.
    n_pred : int
        Predicted item count.
    n_exp : int
        Expected item count.

    Returns
    -------
    tuple of float
        ``(precision, recall, f1)``; two empty sets score a perfect 1.0.
    """
    if n_pred == 0 and n_exp == 0:
        return 1.0, 1.0, 1.0
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_exp if n_exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _pair_f1(predicted: list[tuple[str, str]], expected: list[tuple[str, str]], threshold: float) -> tuple[float, float, float]:
    """Set F1 over (person, label) pairs with fuzzy person matching.

    Parameters
    ----------
    predicted : list of tuple
        Predicted ``(person, label)`` pairs.
    expected : list of tuple
        Expected pairs.
    threshold : float
        Person similarity threshold (0-100).

    Returns
    -------
    tuple of float
        ``(precision, recall, f1)``.

    Notes
    -----
    Labels are compared after text normalization; persons are compared with
    the entity matching of :func:`match_names`, restricted to equal labels.
    """
    used_e: set[int] = set()
    matched = 0
    p_norm = [normalize_person_name(p) for p, _ in predicted]
    e_norm = [normalize_person_name(e) for e, _ in expected]
    for i, (_, p_label) in enumerate(predicted):
        best: Optional[tuple[float, int]] = None
        for j, (_, e_label) in enumerate(expected):
            if j in used_e or _normalize_text(p_label) != _normalize_text(e_label):
                continue
            score = person_similarity(p_norm[i], e_norm[j])
            if score >= threshold and (best is None or score > best[0]):
                best = (score, j)
        if best is not None:
            used_e.add(best[1])
            matched += 1
    return _prf(matched, len(predicted), len(expected))


class Evaluator:
    """Compare pipeline outputs against the ground truth.

    Parameters
    ----------
    settings : Settings
        Pipeline settings (paths and fuzzy threshold).
    repository : BaseStorageRepository or None, optional
        Storage used to load outputs; defaults to a filesystem repository.

    Notes
    -----
    The report is written to ``outputs/evaluation_report.json``.
    """

    def __init__(self, settings: Settings, repository: BaseStorageRepository | None = None) -> None:
        """Store collaborators.

        Parameters
        ----------
        settings : Settings
            Pipeline settings.
        repository : BaseStorageRepository or None, optional
            Output storage; defaults to :class:`FileSystemRepository`.
        """
        self._settings = settings
        self._repository: BaseStorageRepository = repository or FileSystemRepository(settings)
        self._threshold = settings.fuzzy_match_threshold

    def _load_ground_truth(self) -> dict[str, dict[str, Any]]:
        """Load and structurally validate the ground truth.

        Returns
        -------
        dict
            Ground truth keyed by acta identifier.

        Raises
        ------
        EvaluationError
            If the file is missing, unreadable or malformed.
        """
        try:
            raw = FileSystemRepository(self._settings).load_ground_truth()
        except EvaluationError:
            raise
        except PipelineError as exc:
            raise EvaluationError(f"ground truth unreadable: {exc}") from exc
        if not raw:
            raise EvaluationError("ground truth is empty")
        for acta_id, entry in raw.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("fields"), dict):
                raise EvaluationError(f"malformed ground truth entry: {acta_id}")
            missing = [name for name in EVIDENCE_FIELDS if name not in entry["fields"]]
            if missing:
                raise EvaluationError(
                    f"ground truth entry {acta_id} lacks fields: {', '.join(missing)}"
                )
        return raw

    def evaluate(self) -> EvaluationReport:
        """Evaluate every acta in the ground truth and persist the report.

        Returns
        -------
        EvaluationReport
            Aggregate report, also written to ``outputs/evaluation_report.json``.

        Raises
        ------
        EvaluationError
            If the ground truth is missing or malformed.
        """
        truth = self._load_ground_truth()
        documents = [self._evaluate_document(acta_id, entry) for acta_id, entry in truth.items()]
        count = len(documents)
        total_fp = sum(d.critical_false_positives for d in documents)
        global_penalty = CRITICAL_FP_PENALTY if total_fp > 0 else 0.0
        overall = max(0.0, sum(d.final_score for d in documents) / count - global_penalty)
        hit_rate = sum(d.evidence_hit_rate for d in documents) / count
        report = EvaluationReport(
            documents=documents,
            overall_score=_r(overall),
            overall_evidence_hit_rate=_r(hit_rate),
            total_critical_false_positives=total_fp,
            generated_at=datetime.now(timezone.utc).isoformat(),
            global_penalty=global_penalty,
        )
        save_json(self._settings.outputs_dir / REPORT_FILENAME, report.model_dump(mode="json"))
        log_event(
            _LOGGER,
            tag=StatusTag.OK,
            event="evaluation_complete",
            stage=STAGE,
            details={
                "documents": count,
                "overall_score": report.overall_score,
                "global_penalty": report.global_penalty,
                "overall_evidence_hit_rate": report.overall_evidence_hit_rate,
                "total_critical_false_positives": report.total_critical_false_positives,
            },
        )
        return report

    def _failed_document(self, acta_id: str, expected_review: list[str], reason: str) -> DocumentEvaluation:
        """Build a zero-score evaluation for an unusable output.

        Parameters
        ----------
        acta_id : str
            Document identifier.
        expected_review : list of str
            Expected human review fields.
        reason : str
            Explanation of the failure.

        Returns
        -------
        DocumentEvaluation
            Evaluation with all fields scored 0.
        """
        scores = [
            _make_score(name, "missing_output", 0.0, False, reason if k == 0 else "no output to evaluate")
            for k, name in enumerate(FIELD_WEIGHTS)
        ]
        log_event(
            _LOGGER, tag=StatusTag.ERROR, event="output_unavailable", stage=STAGE,
            job_id=acta_id, details={"acta_id": acta_id, "reason": reason},
        )
        return DocumentEvaluation(
            acta_id=acta_id,
            field_scores=scores,
            weighted_score=0.0,
            penalty=0.0,
            final_score=0.0,
            evidence_hit_rate=0.0,
            critical_false_positives=0,
            human_review_expected=sorted(expected_review),
            human_review_flagged=[],
            human_review_match=not expected_review,
        )

    def _evaluate_document(self, acta_id: str, truth: dict[str, Any]) -> DocumentEvaluation:
        """Evaluate one document.

        Parameters
        ----------
        acta_id : str
            Document identifier.
        truth : dict
            Ground truth entry.

        Returns
        -------
        DocumentEvaluation
            Scored document.
        """
        expected_review = [str(x) for x in truth.get("expected_human_review_fields", [])]
        try:
            payload = self._repository.load_output(acta_id)
        except PipelineError as exc:
            return self._failed_document(acta_id, expected_review, f"output unreadable: {exc}")
        if payload is None:
            return self._failed_document(acta_id, expected_review, f"output file outputs/{acta_id}.json not found")
        try:
            output = CorporateDeedOutput.model_validate(payload)
        except PydanticValidationError as exc:
            return self._failed_document(acta_id, expected_review, f"output schema invalid: {exc.error_count()} errors")

        fields: dict[str, Any] = truth["fields"]
        scores = [
            self._score_fecha(output, fields),
            self._score_escritura(output, fields),
            self._score_capital(output, fields),
            self._score_razon(output, fields),
            self._score_notario(output, fields),
            self._score_socios(output, fields),
            self._score_reps(output, fields),
            self._score_poderes(output, fields),
        ]
        weighted = sum(s.weighted for s in scores)
        fp_names = self._critical_false_positives(output, [str(n) for n in truth.get("revoked_representatives", [])])
        penalty = CRITICAL_FP_PENALTY if fp_names else 0.0
        final = max(0.0, weighted - penalty)
        hit_rate = self._evidence_hit_rate(output, fields)
        flagged = sorted(n for n in EVIDENCE_FIELDS if getattr(output, n).requires_human_review)
        expected_sorted = sorted(expected_review)
        evaluation = DocumentEvaluation(
            acta_id=acta_id,
            field_scores=scores,
            weighted_score=_r(weighted),
            penalty=penalty,
            final_score=_r(final),
            evidence_hit_rate=_r(hit_rate),
            critical_false_positives=len(fp_names),
            critical_false_positive_names=fp_names,
            human_review_expected=expected_sorted,
            human_review_flagged=flagged,
            human_review_match=flagged == expected_sorted,
        )
        doc_sha = output.audit_metadata.doc_sha256
        if fp_names:
            log_event(
                _LOGGER, tag=StatusTag.CRITICAL_RISK, event="critical_false_positive", stage=STAGE,
                job_id=acta_id, doc_sha256=doc_sha,
                details={"acta_id": acta_id, "names": fp_names, "penalty": penalty},
            )
        log_event(
            _LOGGER, tag=StatusTag.OK, event="document_evaluated", stage=STAGE,
            job_id=acta_id, doc_sha256=doc_sha,
            details={
                "acta_id": acta_id,
                "weighted_score": evaluation.weighted_score,
                "final_score": evaluation.final_score,
                "evidence_hit_rate": evaluation.evidence_hit_rate,
                "critical_false_positives": evaluation.critical_false_positives,
                "human_review_match": evaluation.human_review_match,
            },
        )
        return evaluation

    def _score_fecha(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score the incorporation date by ISO exact match.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.
        """
        expected = str(fields["fecha_constitucion"]["value"])
        got = out.fecha_constitucion.value
        ok = got is not None and got.strip() == expected
        return _make_score("fecha_constitucion", "iso_exact", 1.0 if ok else 0.0, ok, f"pred={got} gt={expected}")

    def _score_escritura(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score the deed number by digits-only exact match.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.
        """
        expected = _digits(str(fields["numero_escritura"]["value"]))
        got = _digits(out.numero_escritura.value or "")
        ok = bool(got) and got == expected
        return _make_score("numero_escritura", "digits_exact", 1.0 if ok else 0.0, ok, f"pred={got or None} gt={expected}")

    def _score_capital(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score the current share capital within a 0.01 tolerance.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.
        """
        expected = float(fields["capital_social_actual"]["value"]["monto_numerico"])
        cap = out.capital_social_actual.value
        if cap is None:
            return _make_score("capital_social_actual", "numeric_tolerance", 0.0, False, f"pred=None gt={expected:.2f}")
        ok = abs(cap.monto_numerico - expected) <= CAPITAL_TOLERANCE
        return _make_score(
            "capital_social_actual", "numeric_tolerance", 1.0 if ok else 0.0, ok,
            f"pred={cap.monto_numerico:.2f} gt={expected:.2f}",
        )

    def _score_razon(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score the company name by token sort similarity.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.

        Notes
        -----
        The raw score is the similarity itself (partial credit below the 0.90
        pass threshold); ``passed`` reflects the threshold.
        """
        expected = str(fields["razon_social"]["value"])
        got = out.razon_social.value
        if got is None:
            return _make_score("razon_social", "token_sort_ratio", 0.0, False, f"pred=None gt={expected}")
        sim = fuzz.token_sort_ratio(_normalize_text(got), _normalize_text(expected)) / 100.0
        return _make_score(
            "razon_social", "token_sort_ratio", sim, sim >= RAZON_SOCIAL_THRESHOLD,
            f"similarity={sim:.4f} threshold={RAZON_SOCIAL_THRESHOLD:.2f}",
        )

    def _score_notario(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score the notary: 0.5 number exact plus 0.5 name similarity.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.
        """
        expected = fields["notario"]["value"]
        got = out.notario.value
        if got is None:
            return _make_score("notario", "notary_combined", 0.0, False, "pred=None")
        number_ok = got.numero_notaria == int(expected["numero_notaria"])
        sim = fuzz.token_sort_ratio(
            normalize_person_name(got.nombre), normalize_person_name(str(expected["nombre"]))
        ) / 100.0
        combined = 0.5 * (1.0 if number_ok else 0.0) + 0.5 * sim
        return _make_score(
            "notario", "notary_combined", combined, number_ok and sim >= RAZON_SOCIAL_THRESHOLD,
            f"number_match={number_ok} name_similarity={sim:.4f}",
        )

    def _score_socios(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score shareholders by set F1 times percentage accuracy.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.

        Notes
        -----
        Names match when normalized similarity is at least the fuzzy
        threshold (85). ``score = F1 * pct_accuracy`` where ``pct_accuracy``
        is the fraction of matched pairs whose percentages differ by at most
        0.01.
        """
        expected = list(fields["socios"]["value"] or [])
        predicted = list(out.socios.value or [])
        pairs = match_names([s.nombre for s in predicted], [str(e["nombre"]) for e in expected], self._threshold)
        precision, recall, f1 = _prf(len(pairs), len(predicted), len(expected))
        if pairs:
            good = sum(
                1 for i, j, _ in pairs
                if abs(predicted[i].porcentaje - float(expected[j]["porcentaje"])) <= PERCENTAGE_TOLERANCE
            )
            pct_acc = good / len(pairs)
        else:
            pct_acc = 1.0 if not predicted and not expected else 0.0
        score = f1 * pct_acc
        return _make_score(
            "socios", "set_f1_pct", score, score >= 1.0 - 1e-9,
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} pct_accuracy={pct_acc:.4f}",
        )

    def _score_reps(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score active legal representatives by set F1.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record with precision, recall and F1 in the detail.
        """
        expected = [str(e["nombre"]) for e in fields["representantes_legales"]["value"] or []]
        predicted = [r.nombre for r in out.representantes_legales.value or []]
        pairs = match_names(predicted, expected, self._threshold)
        precision, recall, f1 = _prf(len(pairs), len(predicted), len(expected))
        return _make_score(
            "representantes_legales", "set_prf", f1, f1 >= 1.0 - 1e-9,
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} pred={len(predicted)} gt={len(expected)}",
        )

    def _score_poderes(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> FieldScore:
        """Score powers of attorney by set F1 over (grantee, type) pairs.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields.

        Returns
        -------
        FieldScore
            Score record.
        """
        expected = [(str(e["otorgado_a"]), str(e["tipo_poder"])) for e in fields["poderes"]["value"] or []]
        predicted = [(p.otorgado_a, p.tipo_poder) for p in out.poderes.value or []]
        precision, recall, f1 = _pair_f1(predicted, expected, self._threshold)
        return _make_score(
            "poderes", "set_f1_pairs", f1, f1 >= 1.0 - 1e-9,
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} pred={len(predicted)} gt={len(expected)}",
        )

    def _critical_false_positives(self, out: CorporateDeedOutput, revoked: list[str]) -> list[str]:
        """Find revoked representatives that remain in the active list.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        revoked : list of str
            Ground truth revoked representative names.

        Returns
        -------
        list of str
            Revoked names matched (similarity at least the threshold) by an
            active predicted representative.
        """
        active = [normalize_person_name(r.nombre) for r in out.representantes_legales.value or []]
        found: list[str] = []
        for name in revoked:
            norm = normalize_person_name(name)
            if any(person_similarity(norm, cand) >= self._threshold for cand in active):
                found.append(name)
        return found

    def _evidence_hit_rate(self, out: CorporateDeedOutput, fields: dict[str, Any]) -> float:
        """Compute the evidence hit rate over extracted fields.

        Parameters
        ----------
        out : CorporateDeedOutput
            Pipeline output.
        fields : dict
            Ground truth fields with expected pages.

        Returns
        -------
        float
            Fraction of extracted fields with the right page and verified
            grounding; 0.0 when no field was extracted.
        """
        extracted = 0
        hits = 0
        for name in EVIDENCE_FIELDS:
            field = getattr(out, name)
            if field.status != FieldStatus.EXTRACTED:
                continue
            extracted += 1
            evidence = field.evidence
            if (
                evidence is not None
                and evidence.grounding_verified is True
                and evidence.page_number == int(fields[name]["page"])
            ):
                hits += 1
        return hits / extracted if extracted else 0.0
