# CLAUDE.md - Project Governance & Engineering Standards

## 1. Project Overview
Corporate Document Automation Pipeline (Mexican "Actas Constitutivas" - Incorporation Deeds & Amendments).
Bank-grade prototype focusing on document ingestion, hybrid native/OCR extraction, temporal entity resolution, deterministic validation, structured audit logging with PII redaction, Red Hat OpenShift container compliance, and quality evaluation.

## 2. Strict Engineering & Code Standards (ZERO AI SLOP)
- **No AI Slop Policy:**
  - STRICTLY NO emojis anywhere in terminal outputs, logs, comments, docstrings, or Markdown documentation.
  - NEVER use bare `print()` statements in application logic. All runtime events must go through the structured logger (`src/core/logger.py`) or the CLI presenter (`src/cli/main.py`).
  - Terminal output must look like a clean, sober enterprise UNIX financial CLI tool using standard status tags: `[INFO]`, `[OK]`, `[WARN]`, `[REVIEW_REQUIRED]`, `[REVOKED]`, `[ERROR]`.
  - No placeholders, `TODO` comments, or `pass` stubs. Every function must be fully implemented.
- **Language:** ALL source code, variable names, class names, function names, comments, logs, and tests MUST be written strictly in **English**.
  - *Exception:* Domain-specific JSON serialization aliases required by the evaluation spec (`razon_social`, `fecha_constitucion`, `numero_escritura`, `notario`, `socios`, `representantes_legales`, `poderes`, `capital_social_actual`) and Spanish legal text inside synthetic test PDFs and study guides.
- **Docstrings:** Every module, class, and function MUST include concise **NumPy-style docstrings** (`Parameters`, `Returns`, `Raises`, `Notes`).
- **Typing:** Strict Python 3.11+ type hints everywhere.
- **Architecture & SOLID Principles:**
  - **S (Single Responsibility):** Separate PDF inspection, OCR extraction, LLM structured extraction, temporal resolution, deterministic validation, structured logging, storage persistence, and evaluation into distinct modules.
  - **O (Open/Closed):** Pipeline must be config/schema-driven (`DocumentRegistry`) so adding new document types requires no core pipeline modification.
  - **L & I (Liskov & Interface Segregation):** Keep provider interfaces lean (`BaseOCRProvider`, `BaseVerifierProvider`, `BaseLLMProvider`, `BaseStorageRepository`).
  - **D (Dependency Inversion):** High-level pipeline orchestrator must depend on abstractions. Support live LLM execution via `OPENAI_API_KEY` and a deterministic pre-verified cache fallback when run without credentials.

## 3. Enterprise Banking Stack & Red Hat OpenShift Compliance (CRITICAL)
- Keep external dependencies minimal, auditable, and enterprise-friendly (`pydantic`, `pymupdf`, `openai`, `rapidfuzz`, `tenacity`, `typer`, `rich`, `reportlab`, `pillow`, `pytest`) pinned via `uv.lock`.
- **Red Hat OpenShift Container Hardening (`Dockerfile` & `deploy/openshift-deployment.yaml`):**
  - OpenShift enforces strict Security Context Constraints (`restricted-v2` SCC) that run containers with an **arbitrary non-root UID** belonging to GID `0` (root group).
  - The `Dockerfile` MUST NOT run as `root`. It must set `chgrp -R 0 /app && chmod -R g=u /app`, set `ENV HOME=/tmp`, and run under `USER 1001`.
  - Include `deploy/openshift-deployment.yaml` demonstrating a production `Deployment`, `SecurityContext` (`runAsNonRoot: true`, `allowPrivilegeEscalation: false`, `seccompProfile: RuntimeDefault`), and `HorizontalPodAutoscaler` (HPA).

## 4. Structured Audit Logging & PII Redaction
- Implement `src/core/logger.py` with dual output:
  1. Sober human-readable console logs (`stdout`).
  2. Structured JSON Lines audit trail (`logs/pipeline_audit.jsonl`).
- Every log record MUST include contextual fields: `timestamp`, `level`, `job_id`, `doc_sha256`, `stage`, `event`, `duration_ms`, and `details`.
- Implement a deterministic `PIIRedactionFilter` (`logging.Filter`) that masks Mexican RFCs (`[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}` -> `PERJ******ABC`), CURPs, and emails before emission.

## 5. Core Domain Rules to Enforce in Code
1. **Idempotency:** Compute `SHA-256` hash on binary ingestion; check cache (`data/cache/{sha256}.json`) before invoking OCR or LLM.
2. **Page-Level Routing:** Use `PyMuPDF` (`fitz`) per page. If clean alphanumeric characters >= 50, extract natively (`ConfidenceSource.NATIVE_PDF`, score `1.0`). Otherwise, route page image to `LocalOCRProvider` (`rapidocr-onnxruntime` or deterministic fallback) and Escalation Tier 2 (`Decisions` predicate verifier with `|2p - 1|` confidence mapping).
3. **Deterministic Page Attribution:** Inject explicit `<page page_number="N">` XML boundaries during extraction and validate with a deterministic Grounding Check that `source_quote` exists in page `N`.
4. **Temporal Resolution (Acta C):** Extract chronological events into `amendment_history`. Resolve scalar fields (`capital_social_actual`) by latest notarized deed date (`fecha_acto`). Resolve person lists (`representantes_legales`) by normalizing names (stripping accents and titles like `LIC.`, `SR.`, `ING.`) and applying `rapidfuzz` similarity (>= 85% threshold) to subtract revoked individuals while retaining both appointment and revocation evidence.
