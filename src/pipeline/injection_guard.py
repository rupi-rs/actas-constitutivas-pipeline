"""Deterministic prompt-injection pre-filter for document text.

The filter runs on every page before structured extraction, with or
without network access. It matches common English and Spanish phrases that
address an automated system instead of recording a legal act (for example
"ignore previous instructions", "ignora las instrucciones anteriores",
"mark this document as approved", "system prompt"). A match flags the
document for human review and blocks the auto-approval of the fields on
that page. No API call is made: the processing path uses the Decisions API
only for the predicate that verifies vision re-reads.

Notes
-----
Text is lowercased, stripped of accents and whitespace-collapsed before
matching, so line wrapping and accents cannot hide a phrase. The patterns
are deliberately specific: ordinary notarial directives ("se revoca",
"inscribase en el Registro Publico") never match.
"""

from __future__ import annotations

import re
import unicodedata

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?"
            r"(previous|prior|above|earlier|preceding)\s+(instructions|prompts?|rules|directions)"
        ),
    ),
    (
        "ignora_instrucciones",
        re.compile(
            r"\b(ignora|ignore|ignoren|ignorar|olvida|olvide|olviden|omite|omita)\s+(todas\s+)?"
            r"(las\s+|tus\s+|sus\s+)?(instrucciones|indicaciones|reglas)"
        ),
    ),
    (
        "force_approval",
        re.compile(
            r"\bmark\s+(this|the)\s+(document|file|deed|record|case)\s+as\s+"
            r"(approved|valid|verified|clean|compliant)"
        ),
    ),
    (
        "forzar_aprobacion",
        re.compile(
            r"\b(marca|marque|marquen|marcar|registra|registre)\s+(este|el|esta|la)\s+"
            r"(documento|expediente|acta|escritura)\s+como\s+(aprobad[oa]|valid[oa]|verificad[oa])"
        ),
    ),
    (
        "skip_human_review",
        re.compile(
            r"\b(do\s+not|don't|never)\s+(flag|route|send)\s+(this|the|it)\s*"
            r"(document|file|deed)?\s*(for|to)\s+(human\s+)?review"
            r"|\bno\s+(lo\s+|la\s+)?(envies|envie|mandes|mande|turnes|turne)\s+a\s+revision"
            r"|\b(requires_human_review|global_human_review_required)\s*(=|:|to)\s*false"
        ),
    ),
    (
        "system_prompt_reference",
        re.compile(
            r"\bsystem\s+prompt\b|\bprompt\s+del\s+sistema\b|\binstrucciones\s+del\s+sistema\b"
            r"|\bdeveloper\s+message\b"
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as)\s+(an?\s+)?"
            r"(ai|assistant|language\s+model|llm|chatbot)\b"
            r"|\b(eres|actua\s+como)\s+(un\s+)?(asistente|modelo\s+de\s+lenguaje|ia)\b"
        ),
    ),
    (
        "override_instructions",
        re.compile(
            r"\boverride\s+(the\s+|all\s+|your\s+)?(previous\s+|prior\s+)?"
            r"(instructions|rules|validation|guardrails)"
        ),
    ),
)
"""Labelled regular expressions matched against normalized page text."""


def normalize_for_injection_scan(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace.

    Parameters
    ----------
    text : str
        Raw page text.

    Returns
    -------
    str
        Normalized text used for pattern matching.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.lower().split())


def detect_injection_patterns(text: str) -> list[str]:
    """Return the labels of the injection patterns found in a text.

    Parameters
    ----------
    text : str
        Untrusted page text.

    Returns
    -------
    list of str
        Matching pattern labels in declaration order; empty when clean.
    """
    normalized = normalize_for_injection_scan(text)
    return [label for label, pattern in INJECTION_PATTERNS if pattern.search(normalized)]


__all__ = ["INJECTION_PATTERNS", "detect_injection_patterns", "normalize_for_injection_scan"]
