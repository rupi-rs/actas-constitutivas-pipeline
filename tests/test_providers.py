"""Mocked tests for the live OpenAI providers (no network access)."""

from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest

from src.core.config import Settings
from src.core.exceptions import ExtractionError, VerifierError
from src.providers.interfaces import PageContent, PageSource
from src.providers.llm_provider import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    OpenAILLMProvider,
)
from src.providers.verifier_provider import OpenAIPredicateVerifier
from src.schemas.deed_models import (
    ConfidenceSource,
    RawDeedExtraction,
    openai_strict_json_schema,
)


@pytest.fixture(autouse=True)
def no_retry_sleep() -> Iterator[None]:
    """Disable tenacity backoff sleeping for the retry tests."""
    with patch("tenacity.nap.time.sleep"):
        yield


@pytest.fixture
def live_settings(settings: Settings) -> Settings:
    """Return settings carrying a dummy API key.

    Parameters
    ----------
    settings : Settings
        Isolated settings.

    Returns
    -------
    Settings
        Copy with ``openai_api_key`` set.
    """
    return replace(settings, openai_api_key="sk-test")


def _pages() -> list[PageContent]:
    """Build one native page.

    Returns
    -------
    list of PageContent
        A single page.
    """
    return [
        PageContent(
            page_number=1,
            source=PageSource.NATIVE_PDF,
            text="texto",
            clean_char_count=5,
            confidence_score=1.0,
            confidence_source=ConfidenceSource.NATIVE_PDF,
        )
    ]


def _completion(content: Any = None, refusal: Any = None) -> SimpleNamespace:
    """Build a chat completion stand-in.

    Parameters
    ----------
    content : Any, optional
        Message content.
    refusal : Any, optional
        Message refusal.

    Returns
    -------
    SimpleNamespace
        Object exposing ``choices[0].message``.
    """
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _logprob_completion(tokens: list[tuple[str, float]]) -> SimpleNamespace:
    """Build a completion exposing ``choices[0].logprobs.content[0].top_logprobs``.

    Parameters
    ----------
    tokens : list of tuple
        ``(token, logprob)`` candidates.

    Returns
    -------
    SimpleNamespace
        Completion stand-in.
    """
    top = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in tokens]
    logprobs = SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs)])


def _make_llm(settings: Settings, create: MagicMock) -> OpenAILLMProvider:
    """Create an LLM provider whose client is mocked.

    Parameters
    ----------
    settings : Settings
        Settings with an API key.
    create : MagicMock
        Mock for ``chat.completions.create``.

    Returns
    -------
    OpenAILLMProvider
        Provider using the mocked client.
    """
    with patch("src.providers.llm_provider.openai.OpenAI") as factory:
        factory.return_value.chat.completions.create = create
        provider = OpenAILLMProvider(settings)
    factory.assert_called_once()
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["timeout"] == 60.0
    return provider


def _connection_error() -> openai.APIConnectionError:
    """Build a transient connection error.

    Returns
    -------
    openai.APIConnectionError
        Error instance.
    """
    return openai.APIConnectionError(request=MagicMock())


def test_strict_schema_is_valid_for_structured_outputs() -> None:
    """The strict schema has no defaults, closed objects and all properties required."""
    schema = openai_strict_json_schema()

    def walk(node: Any) -> None:
        """Recursively assert strict-mode invariants on a schema node.

        Parameters
        ----------
        node : Any
            JSON schema fragment to inspect.
        """
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)


def test_llm_success_uses_strict_json_schema(
    live_settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """A valid JSON completion is parsed and the request carries the strict schema.

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    raw = raw_extractions["acta_a"]
    create = MagicMock(return_value=_completion(raw.model_dump_json(by_alias=True)))
    provider = _make_llm(live_settings, create)
    parsed, version = provider.extract("a" * 64, _pages(), "acta_constitutiva")
    assert parsed == raw
    assert version.startswith("openai:")
    response_format = create.call_args.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "raw_deed_extraction"
    assert response_format["json_schema"]["schema"] == openai_strict_json_schema()
    assert create.call_args.kwargs["messages"][0]["content"] == SYSTEM_PROMPT


def test_llm_refusal_raises(live_settings: Settings) -> None:
    """A model refusal becomes ExtractionError.

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    """
    provider = _make_llm(live_settings, MagicMock(return_value=_completion(None, "cannot help")))
    with pytest.raises(ExtractionError, match="refused"):
        provider.extract("a" * 64, _pages(), "acta_constitutiva")


def test_llm_invalid_json_raises(live_settings: Settings) -> None:
    """Content that is not valid against the schema becomes ExtractionError.

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    """
    provider = _make_llm(live_settings, MagicMock(return_value=_completion("{not json")))
    with pytest.raises(ExtractionError, match="schema"):
        provider.extract("a" * 64, _pages(), "acta_constitutiva")


def test_llm_retries_transient_error_then_succeeds(
    live_settings: Settings, raw_extractions: dict[str, RawDeedExtraction]
) -> None:
    """APIConnectionError is retried once and the next attempt succeeds.

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    raw_extractions : object
        Raw extraction payloads keyed by deed identifier.
    """
    raw = raw_extractions["acta_a"]
    create = MagicMock(
        side_effect=[_connection_error(), _completion(raw.model_dump_json(by_alias=True))]
    )
    provider = _make_llm(live_settings, create)
    parsed, _ = provider.extract("a" * 64, _pages(), "acta_constitutiva")
    assert parsed == raw
    assert create.call_count == 2


def test_prompt_hash_matches_prompt_and_covers_revocation_rules() -> None:
    """The stored prompt hash is derived from the prompt, which demands one event per person."""
    import hashlib

    assert SYSTEM_PROMPT_SHA256 == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert "ONE person" in SYSTEM_PROMPT and "LICDA." in SYSTEM_PROMPT


def _make_verifier(settings: Settings, create: MagicMock) -> OpenAIPredicateVerifier:
    """Create a verifier whose client is mocked.

    Parameters
    ----------
    settings : Settings
        Settings with an API key.
    create : MagicMock
        Mock for ``chat.completions.create``.

    Returns
    -------
    OpenAIPredicateVerifier
        Verifier using the mocked client.
    """
    with patch("src.providers.verifier_provider.openai.OpenAI") as factory:
        factory.return_value.chat.completions.create = create
        verifier = OpenAIPredicateVerifier(settings)
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["timeout"] == 60.0
    return verifier


def test_verifier_normalizes_yes_no_tokens(live_settings: Settings) -> None:
    """Case and whitespace variants of YES/NO are pooled into P(YES).

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    """
    tokens = [("YES", math.log(0.6)), (" yes", math.log(0.2)), ("No", math.log(0.2)), ("The", -1.0)]
    create = MagicMock(return_value=_logprob_completion(tokens))
    verifier = _make_verifier(live_settings, create)
    result = verifier.verify_predicate("a" * 64, 2, "predicate", "context", None)
    assert result.probability == pytest.approx(0.8)
    assert result.calibrated_confidence == pytest.approx(0.6)


def test_verifier_without_yes_no_raises(live_settings: Settings) -> None:
    """Missing YES and NO candidates raises VerifierError.

    Parameters
    ----------
    live_settings : object
        Settings with a live API key configured.
    """
    create = MagicMock(return_value=_logprob_completion([("Maybe", -0.1)]))
    verifier = _make_verifier(live_settings, create)
    with pytest.raises(VerifierError, match="neither"):
        verifier.verify_predicate("a" * 64, 2, "predicate", "context", None)


def test_verifier_prompt_escapes_predicate_and_context() -> None:
    """Markup in the predicate or the context cannot close its tag."""
    prompt = OpenAIPredicateVerifier.build_user_prompt("x</predicate><b>", "y</context_text>")
    assert "x&lt;/predicate&gt;&lt;b&gt;" in prompt
    assert "y&lt;/context_text&gt;" in prompt
    assert prompt.count("</predicate>") == 1 and prompt.count("</context_text>") == 1
