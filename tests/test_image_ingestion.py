"""Image ingestion tests: PNG/JPG inputs follow the scanned-page route."""

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from PIL import Image
from typer.testing import CliRunner

from src.cli.main import app
from src.core.config import Settings
from src.core.exceptions import CacheMissError, IngestionError
from src.ingestion.hasher import compute_sha256
from src.ingestion.image_loader import (
    images_to_pdf_bytes,
    is_image_path,
    load_document_bytes,
)
from src.ingestion.pdf_inspector import PDFInspector
from src.pipeline.orchestrator import build_orchestrator
from src.providers.interfaces import OCRRegion, OCRResult, PageSource
from src.schemas.deed_models import BoundingBox
from src.storage.repository import FileSystemRepository
from tests.conftest import read_audit_records


def _image_bytes(
    fmt: str, size: tuple[int, int] = (120, 80), color: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    """Encode a solid-colour image.

    Parameters
    ----------
    fmt : str
        Pillow format name (``PNG`` or ``JPEG``).
    size : tuple of (int, int), default (120, 80)
        Width and height in pixels.
    color : tuple of (int, int, int), default (200, 30, 30)
        RGB fill colour.

    Returns
    -------
    bytes
        Encoded image file.
    """
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def test_is_image_path_is_case_insensitive() -> None:
    """Supported suffixes match regardless of case; PDFs do not."""
    assert is_image_path(Path("a.PNG"))
    assert is_image_path(Path("a.jpeg"))
    assert is_image_path(Path("a.jpg"))
    assert not is_image_path(Path("a.pdf"))


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_single_image_wraps_as_one_scanned_page(fmt: str, settings: Settings) -> None:
    """A single image becomes a 1-page PDF without native text."""
    pdf = images_to_pdf_bytes([_image_bytes(fmt)])
    pages = PDFInspector(settings).inspect(pdf)
    assert len(pages) == 1
    assert pages[0].source is PageSource.SCANNED_PAGE
    assert pages[0].clean_char_count == 0
    assert pages[0].image_png is not None


def test_page_size_follows_200_dpi() -> None:
    """Page size in points equals pixels * 72 / 200."""
    pdf = images_to_pdf_bytes([_image_bytes("PNG", (1700, 2200))])
    document = pymupdf.open(stream=pdf, filetype="pdf")  # type: ignore[no-untyped-call]
    try:
        rect = document.load_page(0).rect  # type: ignore[no-untyped-call]
        assert (round(rect.width), round(rect.height)) == (612, 792)
    finally:
        document.close()  # type: ignore[no-untyped-call]


def test_two_images_keep_order(settings: Settings) -> None:
    """Two images yield two pages in the given order."""
    red = _image_bytes("PNG", (100, 100), (255, 0, 0))
    blue = _image_bytes("PNG", (100, 100), (0, 0, 255))
    pdf = images_to_pdf_bytes([red, blue])
    document = pymupdf.open(stream=pdf, filetype="pdf")  # type: ignore[no-untyped-call]
    try:
        assert document.page_count == 2
        first = document.load_page(0).get_pixmap(dpi=72)  # type: ignore[no-untyped-call]
        second = document.load_page(1).get_pixmap(dpi=72)  # type: ignore[no-untyped-call]
        assert first.pixel(10, 10)[0] > 200
        assert second.pixel(10, 10)[2] > 200
    finally:
        document.close()  # type: ignore[no-untyped-call]
    pages = PDFInspector(settings).inspect(pdf)
    assert [page.page_number for page in pages] == [1, 2]


def test_pdf_output_is_deterministic() -> None:
    """Wrapping the same image twice yields identical bytes."""
    data = _image_bytes("PNG")
    assert images_to_pdf_bytes([data]) == images_to_pdf_bytes([data])


def test_single_image_hash_is_raw_sha256(tmp_path: Path) -> None:
    """The hash input of one image is its raw file content."""
    path = tmp_path / "scan.png"
    path.write_bytes(_image_bytes("PNG"))
    original, pdf = load_document_bytes([path])
    assert original == path.read_bytes()
    assert compute_sha256(original) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pdf.startswith(b"%PDF")


def test_multi_image_hash_input_is_length_prefixed(tmp_path: Path) -> None:
    """Several images hash as a length-prefixed concatenation in order."""
    first, second = tmp_path / "a.png", tmp_path / "b.jpg"
    first.write_bytes(_image_bytes("PNG"))
    second.write_bytes(_image_bytes("JPEG"))
    original, _ = load_document_bytes([first, second])
    expected = b"".join(
        struct.pack(">Q", len(blob)) + blob for blob in (first.read_bytes(), second.read_bytes())
    )
    assert original == expected
    reverse, _ = load_document_bytes([second, first])
    assert compute_sha256(reverse) != compute_sha256(original)


def test_corrupt_image_raises_ingestion_error(tmp_path: Path) -> None:
    """A file with an image suffix but garbage content is rejected."""
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image at all")
    with pytest.raises(IngestionError):
        load_document_bytes([path])
    with pytest.raises(IngestionError):
        images_to_pdf_bytes([b"\x89PNG\r\n\x1a\ntruncated"])


def test_unsupported_suffix_and_empty_inputs(tmp_path: Path) -> None:
    """Unsupported suffixes, empty files and empty lists are rejected."""
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a")
    empty = tmp_path / "e.png"
    empty.write_bytes(b"")
    for bad in ([gif], [empty], []):
        with pytest.raises(IngestionError):
            load_document_bytes(bad)
    with pytest.raises(IngestionError):
        images_to_pdf_bytes([])


def test_exif_orientation_is_applied() -> None:
    """A landscape pixel grid tagged orientation 6 becomes a portrait page."""
    image = Image.new("RGB", (300, 100), (10, 120, 10))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    pdf = images_to_pdf_bytes([buffer.getvalue()])
    document = pymupdf.open(stream=pdf, filetype="pdf")  # type: ignore[no-untyped-call]
    try:
        rect = document.load_page(0).rect  # type: ignore[no-untyped-call]
        assert rect.height > rect.width
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _image_document(
    settings: Settings, cache_payload: dict[str, Any], acta_pdf: Path, tmp_path: Path
) -> tuple[list[Path], str]:
    """Render a sample PDF to PNG pages and register a matching cache entry.

    Parameters
    ----------
    settings : Settings
        Isolated settings.
    cache_payload : dict
        Pre-verified cache payload of the sample.
    acta_pdf : Path
        Sample PDF to rasterise.
    tmp_path : Path
        Directory for the image files.

    Returns
    -------
    tuple of (list of Path, str)
        Image paths in page order and the SHA-256 of the original bytes.
    """
    document = pymupdf.open(stream=acta_pdf.read_bytes(), filetype="pdf")  # type: ignore[no-untyped-call]
    paths: list[Path] = []
    ocr_pages: dict[str, Any] = dict(cache_payload["ocr_pages"])
    try:
        for index in range(document.page_count):
            page = document.load_page(index)  # type: ignore[no-untyped-call]
            path = tmp_path / f"scan_{index + 1}.png"
            path.write_bytes(page.get_pixmap(dpi=200).tobytes("png"))
            paths.append(path)
            key = str(index + 1)
            if key not in ocr_pages:
                bounds = page.rect
                text = str(page.get_text("text"))
                ocr_pages[key] = OCRResult(
                    page_number=index + 1,
                    text=text,
                    mean_confidence=0.99,
                    regions=[
                        OCRRegion(
                            text=text,
                            bbox=BoundingBox(x1=0.0, y1=0.0, x2=bounds.width, y2=bounds.height),
                            confidence=0.99,
                        )
                    ],
                    engine="test-ocr",
                ).model_dump(mode="json")
    finally:
        document.close()  # type: ignore[no-untyped-call]
    original, _ = load_document_bytes(paths)
    sha = compute_sha256(original)
    payload = {**cache_payload, "doc_sha256": sha, "ocr_pages": ocr_pages}
    FileSystemRepository(settings).save_cache(sha, payload)
    return paths, sha


def test_orchestrator_cache_hit_for_image_document(
    settings: Settings,
    sample_paths: dict[str, Path],
    cache_payloads: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Images of a deed served from a cache keep evidence on the right page."""
    orchestrator = build_orchestrator(settings, live=False)
    reference = orchestrator.process_document(sample_paths["acta_b"])
    paths, sha = _image_document(
        settings, cache_payloads["acta_b"], sample_paths["acta_b"], tmp_path
    )
    result = orchestrator.process_images(paths, acta_id="acta_b_image")
    assert result.cache_hit
    assert result.doc_sha256 == sha
    meta = result.output.audit_metadata
    assert meta.total_pages == len(paths)
    assert meta.native_pages_count == 0
    assert meta.ocr_pages_count == len(paths)
    assert meta.document_filename == ", ".join(path.name for path in paths)
    expected = reference.output.razon_social.evidence
    actual = result.output.razon_social.evidence
    assert expected is not None and actual is not None
    assert actual.page_number == expected.page_number
    assert result.output.capital_social_actual.value == reference.output.capital_social_actual.value
    assert result.output_path.name == "acta_b_image.json"


def test_process_document_logs_image_source_kind(
    settings: Settings, audit_log_path: Path, tmp_path: Path
) -> None:
    """A single uncached image logs source_kind=image, then a cache miss."""
    image = tmp_path / "single.jpg"
    image.write_bytes(_image_bytes("JPEG"))
    orchestrator = build_orchestrator(settings, live=False)
    with pytest.raises(CacheMissError):
        orchestrator.process_document(image)
    records = read_audit_records(audit_log_path)
    ingested = [r for r in records if r["event"] == "document_ingested"]
    assert len(ingested) == 1
    assert ingested[0]["stage"] == "ingestion"
    assert ingested[0]["details"]["source_kind"] == "image"
    assert ingested[0]["details"]["image_count"] == 1
    assert ingested[0]["doc_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_mixed_pdf_and_image_is_rejected(
    settings: Settings, sample_paths: dict[str, Path], tmp_path: Path
) -> None:
    """A document cannot combine a PDF with images."""
    image = tmp_path / "x.png"
    image.write_bytes(_image_bytes("PNG"))
    orchestrator = build_orchestrator(settings, live=False)
    with pytest.raises(IngestionError):
        orchestrator.process_files([sample_paths["acta_a"], image])


def test_cli_process_file_without_cache_fails_cleanly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An uncached image exits 1 with the CacheMissError message."""
    monkeypatch.setenv("ACTAS_PROJECT_ROOT", str(settings.project_root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    image = tmp_path / "unknown.png"
    image.write_bytes(_image_bytes("PNG"))
    result = CliRunner().invoke(app, ["process-file", str(image)])
    assert result.exit_code == 1
    assert "CacheMissError" in result.output
    assert "no cache entry for unknown" in result.output


def test_cli_process_file_serves_image_from_cache(
    settings: Settings,
    sample_paths: dict[str, Path],
    cache_payloads: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """process-file prints the summary row for a cached image document."""
    paths, _ = _image_document(
        settings, cache_payloads["acta_b"], sample_paths["acta_b"], tmp_path
    )
    monkeypatch.setenv("ACTAS_PROJECT_ROOT", str(settings.project_root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = ["process-file", *[str(path) for path in paths], "--acta-id", "acta_b_image"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "acta_b_image" in result.output
    assert "[OK]" in result.output
