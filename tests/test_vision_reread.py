"""VLM re-read tests: provider request shape, review routing, crops, cache and base URL."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src.core.config import Settings, load_settings_from_env
from src.core.exceptions import VisionReadError
from src.generators.deed_content import ACTA_B_CAPITAL_BBOX, B_P2_CAPITAL
from src.ingestion.hasher import compute_file_sha256
from src.pipeline.orchestrator import ProviderBundle, build_orchestrator
from src.providers.interfaces import (
    BaseLLMProvider,
    BaseVisionReader,
    PageContent,
    VisionReadResult,
)
from src.providers.llm_provider import CachedLLMProvider, OpenAILLMProvider
from src.providers.ocr_provider import DeterministicOCRProvider
from src.providers.verifier_provider import (
    DeterministicVerifierProvider,
    OpenAIDecisionsVerifier,
)
from src.providers.vision_provider import (
    VISION_READ_SCHEMA,
    DeterministicVLMProvider,
    OpenAIVisionReader,
)
from src.schemas.deed_models import (
    PIPELINE_EVIDENCE_FIELDS,
    BoundingBox,
    RawDeedExtraction,
    openai_strict_json_schema,
)
from src.storage.repository import FileSystemRepository
from tests.conftest import read_audit_records
from tests.test_idempotency import SpyLLM, _StaticRepository, build_custom_orchestrator

SPEC_REASON: str = (
    "Low OCR confidence (0.72) and Decisions predicate confidence (0.56 < 0.80) due to "
    "overlapping notary seal on page 2 bbox [72.0, 410.0, 520.0, 455.0]."
)
"""Spec-02 review reason of the Acta B capital clause."""

MISREAD: str = "QUINTA.- El capital social es de $25O,OOO.0O (DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.)."
"""Simulated Tier 1 OCR misreading of the sealed capital clause."""

CROP_PATH: str = "outputs/crops/acta_b_p2_capital_social_actual_bbox.png"
"""Expected project-relative crop location for the Acta B capital evidence."""


class SpyVision(BaseVisionReader):
    """Vision reader delegating to another reader while recording calls."""

    def __init__(self, delegate: BaseVisionReader) -> None:
        """Store the delegate.

        Parameters
        ----------
        delegate : BaseVisionReader
            Reader that actually serves the re-read.
        """
        self._delegate = delegate
        self.calls: list[tuple[int, list[float], Optional[bytes], str]] = []

    def reread_region(
        self,
        doc_sha256: str,
        page_number: int,
        bbox: BoundingBox,
        image_png: Optional[bytes],
        ocr_text: str,
    ) -> VisionReadResult:
        """Record the call and delegate.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            One-based page number.
        bbox : BoundingBox
            Region box.
        image_png : bytes or None
            Region crop.
        ocr_text : str
            OCR reading.

        Returns
        -------
        VisionReadResult
            Delegate result.
        """
        self.calls.append((page_number, bbox.as_list(), image_png, ocr_text))
        return self._delegate.reread_region(doc_sha256, page_number, bbox, image_png, ocr_text)


class FailingVision(BaseVisionReader):
    """Vision reader that always fails as a refusal would."""

    def reread_region(
        self,
        doc_sha256: str,
        page_number: int,
        bbox: BoundingBox,
        image_png: Optional[bytes],
        ocr_text: str,
    ) -> VisionReadResult:
        """Raise the refusal error.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        page_number : int
            One-based page number.
        bbox : BoundingBox
            Region box.
        image_png : bytes or None
            Region crop.
        ocr_text : str
            OCR reading.

        Returns
        -------
        VisionReadResult
            Never returns.

        Raises
        ------
        VisionReadError
            Always.
        """
        raise VisionReadError("vision model refused to re-read the region")


class RecordingLLM(BaseLLMProvider):
    """LLM provider recording the page texts it receives."""

    def __init__(self, delegate: BaseLLMProvider) -> None:
        """Store the delegate.

        Parameters
        ----------
        delegate : BaseLLMProvider
            Provider that serves the extraction.
        """
        self._delegate = delegate
        self.pages: list[PageContent] = []

    def extract(
        self, doc_sha256: str, pages: list[PageContent], document_type: str
    ) -> tuple[RawDeedExtraction, str]:
        """Record the pages and delegate.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        pages : list of PageContent
            Page contents.
        document_type : str
            Registered document type.

        Returns
        -------
        tuple of (RawDeedExtraction, str)
            Delegate result.
        """
        self.pages = list(pages)
        return self._delegate.extract(doc_sha256, pages, document_type)


def _fake_client(response: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Build a fake OpenAI client whose ``responses.create`` records kwargs.

    Parameters
    ----------
    response : Any
        Object returned by ``responses.create``.

    Returns
    -------
    tuple of (Any, list of dict)
        Fake client and the list of recorded call kwargs.
    """
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        """Record the request and return the canned response.

        Parameters
        ----------
        **kwargs : Any
            Request keyword arguments.

        Returns
        -------
        Any
            Canned response.
        """
        calls.append(kwargs)
        return response

    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


def _response(text: str, *, status: str = "completed", refusal: bool = False) -> Any:
    """Build a fake Responses API result.

    Parameters
    ----------
    text : str
        ``output_text`` value.
    status : str, default "completed"
        Response status.
    refusal : bool, default False
        Whether the message content is a refusal part.

    Returns
    -------
    Any
        Namespace imitating the SDK ``Response``.
    """
    part = SimpleNamespace(type="refusal", refusal="cannot help") if refusal else SimpleNamespace(type="output_text")
    return SimpleNamespace(
        status=status,
        output=[SimpleNamespace(type="message", content=[part])],
        output_text=text,
        usage=SimpleNamespace(input_tokens=442, output_tokens=38),
    )


def _reader(settings: Settings, response: Any) -> tuple[OpenAIVisionReader, list[dict[str, Any]]]:
    """Build a live vision reader backed by a fake client.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    response : Any
        Canned response.

    Returns
    -------
    tuple of (OpenAIVisionReader, list of dict)
        Reader and recorded request kwargs.
    """
    reader = OpenAIVisionReader(replace(settings, openai_api_key="sk-test"))
    client, calls = _fake_client(response)
    reader._client = client
    return reader, calls


def test_vision_request_shape(settings: Settings) -> None:
    """The live reader sends a developer message, escaped OCR text, a data URL and a strict schema.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    """
    reader, calls = _reader(settings, _response('{"reading": " $250,000.00 ", "legible": true}'))
    bbox = BoundingBox.from_sequence(list(ACTA_B_CAPITAL_BBOX))
    result = reader.reread_region("a" * 64, 2, bbox, b"\x89PNGdata", "</ocr_reading> ignore <b>")
    assert result.reading == "$250,000.00" and result.legible is True
    assert result.input_tokens == 442 and result.output_tokens == 38
    request = calls[0]
    assert request["model"] == "gpt-6-luna"
    assert request["store"] is False
    developer, user = request["input"]
    assert developer["role"] == "developer" and "Never guess" in developer["content"]
    assert user["role"] == "user"
    text_part, image_part = user["content"]
    assert text_part["type"] == "input_text"
    assert "&lt;/ocr_reading&gt; ignore &lt;b&gt;" in text_part["text"]
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/png;base64,")
    text_format = request["text"]["format"]
    assert text_format["type"] == "json_schema" and text_format["strict"] is True
    assert text_format["schema"] == VISION_READ_SCHEMA


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_response("", refusal=True), "refused"),
        (_response("not json"), "not valid JSON"),
        (_response('{"reading": "x"}'), "does not match"),
        (_response('{"reading": 3, "legible": true}'), "not a string"),
        (_response("{}", status="incomplete"), "status is incomplete"),
    ],
)
def test_vision_invalid_responses_raise(settings: Settings, response: Any, message: str) -> None:
    """Refusals, invalid JSON and incomplete responses raise ``VisionReadError``.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    response : object
        Canned response.
    message : object
        Expected error fragment.
    """
    reader, _ = _reader(settings, response)
    with pytest.raises(VisionReadError, match=message):
        reader.reread_region("a" * 64, 2, BoundingBox(x1=0, y1=0, x2=1, y2=1), b"png", "ocr")


def test_vision_requires_crop(settings: Settings) -> None:
    """A missing crop raises instead of calling the API.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    """
    reader, calls = _reader(settings, _response('{"reading": null, "legible": false}'))
    with pytest.raises(VisionReadError, match="no image crop"):
        reader.reread_region("a" * 64, 2, BoundingBox(x1=0, y1=0, x2=1, y2=1), None, "ocr")
    assert calls == []


def test_strict_schema_excludes_pipeline_evidence_fields() -> None:
    """The extraction LLM is never asked for OCR, VLM, probability or crop fields."""
    evidence = openai_strict_json_schema()["$defs"]["Evidence"]
    for name in PIPELINE_EVIDENCE_FIELDS:
        assert name not in evidence["properties"]
        assert name not in evidence["required"]


def _acta_b_cache(settings: Settings, sample_paths: dict[str, Path]) -> tuple[Path, dict[str, Any]]:
    """Return the Acta B cache file and its parsed payload.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    sample_paths : dict of str to Path
        Sample PDFs.

    Returns
    -------
    tuple of (Path, dict)
        Cache path and payload.
    """
    path = settings.cache_file_for(compute_file_sha256(sample_paths["acta_b"]))
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rewrite_acta_b(
    settings: Settings,
    sample_paths: dict[str, Path],
    *,
    ocr_text: str,
    reading: Optional[str],
    legible: bool,
    probability: float,
) -> None:
    """Rewrite the Acta B cache with a custom OCR text, VLM read and Decisions p.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    sample_paths : dict of str to Path
        Sample PDFs.
    ocr_text : str
        Tier 1 OCR text of the capital region.
    reading : str or None
        Cached VLM reading.
    legible : bool
        Cached VLM legibility verdict.
    probability : float
        Cached Decisions probability.
    """
    path, payload = _acta_b_cache(settings, sample_paths)
    page = payload["ocr_pages"]["2"]
    for region in page["regions"]:
        if region["text"] == B_P2_CAPITAL:
            region["text"] = ocr_text
    page["text"] = page["text"].replace(B_P2_CAPITAL, ocr_text)
    payload["vlm_reads"][0].update({"reading": reading, "legible": legible})
    payload["verifier_responses"][0]["probability"] = probability
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _deterministic_orchestrator(
    settings: Settings,
) -> tuple[Any, SpyVision, RecordingLLM]:
    """Build a cache-backed orchestrator with spy vision and recording LLM.

    Parameters
    ----------
    settings : Settings
        Isolated settings.

    Returns
    -------
    tuple
        Orchestrator, vision spy and recording LLM.
    """
    repository = FileSystemRepository(settings)
    vision = SpyVision(DeterministicVLMProvider(repository))
    llm = RecordingLLM(CachedLLMProvider(repository))
    bundle = ProviderBundle(
        ocr=DeterministicOCRProvider(repository),
        verifier=DeterministicVerifierProvider(repository),
        llm=llm,
        vision=vision,
    )
    return build_custom_orchestrator(settings, bundle, None), vision, llm


def test_deterministic_acta_b_reason_is_exact_and_crop_written(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """The default build keeps the spec-02 reason byte-exact and persists the crop.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    result = build_orchestrator(settings, live=False).process_document(sample_paths["acta_b"])
    capital = result.output.capital_social_actual
    assert capital.requires_human_review is True
    assert capital.review_reason == SPEC_REASON
    evidence = capital.evidence
    assert evidence is not None
    assert evidence.ocr_reading == B_P2_CAPITAL and evidence.vlm_reading == B_P2_CAPITAL
    assert evidence.vlm_legible is True
    assert evidence.verification_probability == pytest.approx(0.78)
    assert evidence.crop_path == CROP_PATH
    crop = settings.project_root / CROP_PATH
    assert crop.is_file() and crop.read_bytes().startswith(b"\x89PNG")


def test_only_low_confidence_scanned_regions_are_reread(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """Acta B re-reads only its 0.72 region; native Actas A and C are never re-read.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    orchestrator, vision, _ = _deterministic_orchestrator(settings)
    for acta_id in ("acta_a", "acta_c"):
        orchestrator.process_document(sample_paths[acta_id])
    assert vision.calls == []
    orchestrator.process_document(sample_paths["acta_b"])
    assert len(vision.calls) == 1
    page_number, bbox, crop, ocr_text = vision.calls[0]
    assert page_number == 2 and bbox == list(ACTA_B_CAPITAL_BBOX)
    assert crop is not None and crop.startswith(b"\x89PNG")
    assert ocr_text == B_P2_CAPITAL


def test_confirmed_vlm_correction_needs_no_review(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A legible correction confirmed by Decisions >= 0.80 is accepted without review.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    _rewrite_acta_b(
        settings, sample_paths, ocr_text=MISREAD, reading=B_P2_CAPITAL, legible=True, probability=0.95
    )
    orchestrator, _, llm = _deterministic_orchestrator(settings)
    result = orchestrator.process_document(sample_paths["acta_b"])
    page_two = next(page for page in llm.pages if page.page_number == 2)
    assert B_P2_CAPITAL in page_two.text and MISREAD not in page_two.text
    capital = result.output.capital_social_actual
    assert capital.requires_human_review is False and capital.review_reason is None
    assert capital.value is not None and capital.value.monto_numerico == pytest.approx(250000.0)
    evidence = capital.evidence
    assert evidence is not None and evidence.grounding_verified is True
    assert evidence.ocr_reading == MISREAD and evidence.vlm_reading == B_P2_CAPITAL
    assert evidence.confidence_score == pytest.approx(0.9)
    records = read_audit_records(settings.audit_log_path)
    confirmed = [r for r in records if r["event"] == "vlm_correction_confirmed"]
    assert confirmed and confirmed[0]["tag"] == "OK"
    assert any(r["event"] == "vlm_correction_applied" for r in records)


def test_illegible_vlm_read_routes_to_review(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """``legible=false`` routes to review even when Decisions is confident.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    _rewrite_acta_b(
        settings, sample_paths, ocr_text=B_P2_CAPITAL, reading=None, legible=False, probability=0.95
    )
    orchestrator, _, _ = _deterministic_orchestrator(settings)
    capital = orchestrator.process_document(sample_paths["acta_b"]).output.capital_social_actual
    assert capital.requires_human_review is True
    reason = capital.review_reason or ""
    assert "legible=false" in reason and "reading null" in reason
    assert "OCR confidence 0.72" in reason and "p=0.95" in reason and "calibrated 0.90" in reason
    assert "[72.0, 410.0, 520.0, 455.0]" in reason and "crops" not in reason
    assert capital.evidence is not None and capital.evidence.vlm_legible is False


def test_disagreement_with_low_decisions_routes_to_review(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """OCR/VLM disagreement plus low Decisions confidence requires review with both facts.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    _rewrite_acta_b(
        settings, sample_paths, ocr_text=MISREAD, reading=B_P2_CAPITAL, legible=True, probability=0.78
    )
    orchestrator, _, _ = _deterministic_orchestrator(settings)
    capital = orchestrator.process_document(sample_paths["acta_b"]).output.capital_social_actual
    assert capital.requires_human_review is True
    reason = capital.review_reason or ""
    assert reason.startswith(SPEC_REASON)
    assert "OCR and VLM readings disagree" in reason and "$25O,OOO.0O" in reason


def test_vision_failure_routes_to_review_and_skips_cache(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A refused re-read keeps the OCR text, flags the field and blocks live persistence.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    settings = replace(settings, openai_extraction_model="fake-live-model")
    path, payload = _acta_b_cache(settings, sample_paths)
    path.unlink()
    live_repo = _StaticRepository(settings, payload)
    live = ProviderBundle(
        ocr=DeterministicOCRProvider(live_repo),
        verifier=DeterministicVerifierProvider(live_repo),
        llm=SpyLLM(CachedLLMProvider(live_repo)),
        vision=FailingVision(),
    )
    orchestrator = build_custom_orchestrator(settings, live, live)
    capital = orchestrator.process_document(sample_paths["acta_b"]).output.capital_social_actual
    assert capital.requires_human_review is True
    assert "VLM re-read unavailable (vision model refused" in (capital.review_reason or "")
    assert not path.exists()
    events = [r["event"] for r in read_audit_records(settings.audit_log_path)]
    assert "vlm_reread_failed" in events and "cache_not_written" in events


def test_live_miss_persists_vlm_reads_round_trip(
    settings: Settings, sample_paths: dict[str, Path]
) -> None:
    """A live miss writes ``vlm_reads`` that the deterministic reader serves back.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    sample_paths : object
        Generated sample PDF paths keyed by deed identifier.
    """
    settings = replace(settings, openai_extraction_model="fake-live-model")
    path, payload = _acta_b_cache(settings, sample_paths)
    path.unlink()
    live_repo = _StaticRepository(settings, payload)
    live = ProviderBundle(
        ocr=DeterministicOCRProvider(live_repo),
        verifier=DeterministicVerifierProvider(live_repo),
        llm=SpyLLM(CachedLLMProvider(live_repo)),
        vision=DeterministicVLMProvider(live_repo),
    )
    repository = FileSystemRepository(settings)
    cached = ProviderBundle(
        ocr=DeterministicOCRProvider(repository),
        verifier=DeterministicVerifierProvider(repository),
        llm=CachedLLMProvider(repository),
        vision=DeterministicVLMProvider(repository),
    )
    orchestrator = build_custom_orchestrator(settings, cached, live)
    first = orchestrator.process_document(sample_paths["acta_b"])
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["vlm_reads"] == payload["vlm_reads"]
    assert persisted["ocr_pages"] == payload["ocr_pages"]
    replayed = DeterministicVLMProvider(repository).reread_region(
        first.doc_sha256, 2, BoundingBox.from_sequence(list(ACTA_B_CAPITAL_BBOX)), None, ""
    )
    assert replayed.reading == B_P2_CAPITAL and replayed.legible is True
    second = orchestrator.process_document(sample_paths["acta_b"])
    assert second.cache_hit is True
    assert second.output.capital_social_actual.review_reason == SPEC_REASON


def test_deterministic_reader_miss_raises(settings: Settings) -> None:
    """A cache entry without a matching re-read raises instead of guessing.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    """
    reader = DeterministicVLMProvider(_StaticRepository(settings, {"vlm_reads": []}))
    with pytest.raises(VisionReadError, match="no cached vision re-read"):
        reader.reread_region("a" * 64, 2, BoundingBox(x1=0, y1=0, x2=1, y2=1), None, "")


def test_base_url_from_env_reaches_every_client(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OPENAI_BASE_URL`` is loaded and passed to the extraction, vision and Decisions clients.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest environment patcher.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.internal/v1")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "custom-vision-model")
    loaded = load_settings_from_env()
    assert loaded.openai_base_url == "https://gateway.example.internal/v1"
    assert loaded.openai_vision_model == "custom-vision-model"
    live = replace(settings, openai_api_key="sk-test", openai_base_url=loaded.openai_base_url)
    vision = OpenAIVisionReader(live)
    assert vision.engine_name == "openai-vision:gpt-6-luna"
    for client in (
        OpenAILLMProvider(live)._client,
        vision._client,
        OpenAIDecisionsVerifier(live)._client,
    ):
        assert str(client.base_url).rstrip("/") == "https://gateway.example.internal/v1"


def test_base_url_unset_keeps_sdk_default(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``OPENAI_BASE_URL`` the clients use the SDK default endpoint.

    Parameters
    ----------
    settings : object
        Isolated workspace settings.
    monkeypatch : object
        Pytest environment patcher.
    """
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
    loaded = load_settings_from_env()
    assert loaded.openai_base_url is None
    assert loaded.openai_vision_model == "gpt-6-luna"
    live = replace(settings, openai_api_key="sk-test")
    assert str(OpenAIVisionReader(live)._client.base_url).rstrip("/") == "https://api.openai.com/v1"
