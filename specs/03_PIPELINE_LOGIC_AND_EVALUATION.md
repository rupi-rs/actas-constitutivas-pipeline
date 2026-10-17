# Specification 03: Pipeline Algorithms, Validation, Evaluation & OpenShift Infra

## 1. Name Normalization & Fuzzy Entity Matching (`src/pipeline/resolver.py`)
Implement `normalize_person_name(raw_name: str) -> str`:
1. Uppercase and strip whitespace.
2. Strip accents using `unicodedata.normalize("NFKD", text)` filtering out `unicodedata.combining(c)`.
3. Remove Mexican professional and courtesy prefixes via regex:
   `r"\b(LICENCIADO|LICENCIADA|LIC|INGENIERO|INGENIERA|ING|SENOR|SENORA|SR|SRA|SRITA|DOCTOR|DOCTORA|DR|DRA|MAESTRO|MAESTRA|MTRO|MTRA|CIUDADANO|CIUDADANA|C)\b\.?"`
4. Remove punctuation (`.,;-`) and collapse multiple spaces.
5. Compute similarity between normalized target and candidate using:
   `score = max(fuzz.token_sort_ratio(norm_a, norm_b), fuzz.token_set_ratio(norm_a, norm_b) * 0.95)`
   Also support initial expansion matching (e.g., `"C"` matching `"CARLOS"` when first token `"JUAN"` and last tokens `"PEREZ LOPEZ"` match 100%, boosting score to `>= 88.0`).
6. If `score >= 85.0`, mark the original representative and their powers as revoked (`vigente = False`), exclude them from active `representantes_legales.value`, and log `[REVOKED]` with the exact similarity score.

---

## 2. Deterministic Validators (`src/pipeline/validators.py`)
1. **Grounding Check (`verify_grounding`):**
   - For every `FieldWithEvidence`, check that `evidence.source_quote` exists inside the extracted text of `evidence.page_number`.
   - Normalize whitespace and accents on both strings; if exact substring match OR `rapidfuzz.fuzz.partial_ratio(norm_quote, norm_page_text) >= 90.0`, set `evidence.grounding_verified = True`.
   - Otherwise, set `evidence.grounding_verified = False`, degrade `confidence_score = min(confidence_score, 0.40)`, set `requires_human_review = True`, and append reason `"Grounding check failed: quote not found on specified page"`.
2. **Number vs. Words Check (`verify_capital_consistency`):**
   - Check that `capital_social_actual.value.monto_numerico` matches standard Spanish legal words in `monto_en_letra` (`500000.0` <-> `"QUINIENTOS MIL"`, `250000.0` <-> `"DOSCIENTOS CINCUENTA MIL"`, `1000000.0` <-> `"UN MILLON"`).
3. **Confidence Threshold Routing:**
   - If any field has `evidence.confidence_score < 0.80`, set `requires_human_review = True`.

---

## 3. Evaluation Metrics & Error Asymmetry (`src/evaluation/evaluator.py`)
Compare `outputs/{acta_id}.json` against `data/ground_truth/ground_truth.json`:
1. **Scalar Exact Match (Normalized):**
   - `fecha_constitucion` (ISO date exact match, weight `0.10`)
   - `numero_escritura` (digits normalized exact match, weight `0.10`)
   - `capital_social_actual` (numeric float exact match within `0.01`, weight `0.20`)
2. **Fuzzy String Similarity (`rapidfuzz.fuzz.token_sort_ratio / 100.0`):**
   - `razon_social` (threshold `>= 0.90`, weight `0.10`)
   - `notario` (combined notary number exact match + name similarity, weight `0.10`)
3. **Entity List Set F1-Score (Matched via normalized name `>= 85%`):**
   - `socios` (Set F1 on shareholder names + share percentage accuracy, weight `0.10`)
   - `representantes_legales` (Set Precision, Recall, and F1 on active representatives, weight `0.20`)
   - `poderes` (Set F1 on active power grants, weight `0.10`)
4. **Critical False Positive Penalty (Banking Risk Asymmetry):**
   - Explicitly check if any revoked representative (`"JUAN CARLOS PEREZ LOPEZ"` in Acta C) appears in the active `representantes_legales` list. Count `critical_false_positives`. If `critical_false_positives > 0`, apply a severe penalty (`-0.25` to global score) and flag `[CRITICAL_RISK]`.
5. **Evidence Hit Rate:**
   - Percentage of extracted fields where `page_number` matches ground truth page AND `grounding_verified is True`.

---

## 4. CLI Commands (`src/cli/main.py`)
Build a `Typer` CLI entry point registered in `pyproject.toml` as `actas-cli = "src.cli.main:app"` with 4 commands:
1. `actas-cli generate-samples`: Generates `acta_a.pdf`, `acta_b.pdf`, `acta_c.pdf`, `ground_truth.json`, pre-warmed SHA-256 caches in `data/cache/`, and the 2 Spanish study PDFs in `study_guides/`.
2. `actas-cli process-all [--live / --no-live]`: Processes all 3 PDFs through the pipeline, saves JSON outputs to `outputs/`, writes audit logs to `logs/pipeline_audit.jsonl`, and displays a sober Rich summary table.
3. `actas-cli evaluate`: Runs `evaluator.py` comparing `outputs/` against `data/ground_truth/ground_truth.json`, saves `outputs/evaluation_report.json`, and prints a detailed Rich table of field metrics, Evidence Hit Rate, and Critical False Positives.
4. `actas-cli demo`: Executes `generate-samples` -> `process-all` -> `evaluate` end-to-end in a single command.
