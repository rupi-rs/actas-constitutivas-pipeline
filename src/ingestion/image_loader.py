"""Image ingestion: wrap PNG/JPG inputs as scanned PDF pages.

A photographed or scanned page delivered as an image has no text layer, so it
must follow the scanned-page route. Wrapping each image as one PDF page lets
the inspector, OCR providers, grounding checks and verification crops run
unchanged, with evidence page numbers equal to the image order.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from src.core.exceptions import IngestionError
from src.ingestion.pdf_inspector import RENDER_DPI

SUPPORTED_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
"""Lowercase file suffixes accepted as image documents."""

_POINTS_PER_INCH: float = 72.0
_ORIENTATION_TAG: int = 0x0112
_NATIVE_MODES: frozenset[str] = frozenset({"1", "L", "RGB", "RGBA"})
_LENGTH_PREFIX = struct.Struct(">Q")


def is_image_path(path: Path) -> bool:
    """Tell whether a path has a supported image suffix.

    Parameters
    ----------
    path : Path
        File path.

    Returns
    -------
    bool
        ``True`` for ``.png``, ``.jpg`` and ``.jpeg`` (case-insensitive).
    """
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _prepare_image(data: bytes, label: str) -> tuple[bytes, tuple[int, int]]:
    """Validate an image and normalise its orientation.

    Parameters
    ----------
    data : bytes
        Raw image file content.
    label : str
        File name used in error messages.

    Returns
    -------
    tuple of (bytes, tuple of (int, int))
        Bytes to embed in the PDF and the pixel size ``(width, height)`` of
        the upright image.

    Raises
    ------
    IngestionError
        If the payload is not a readable PNG or JPEG image.

    Notes
    -----
    The EXIF orientation is applied with ``ImageOps.exif_transpose`` so a
    phone photo stored sideways is read upright by OCR. When the pixels must
    change (rotation applied, or a colour mode PDF embedding does not take
    directly) the upright image is re-encoded losslessly as PNG; otherwise the
    original bytes are embedded untouched.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            if image_format not in ("PNG", "JPEG"):
                raise IngestionError(
                    f"unsupported image format in {label}: {image_format}",
                    {"file": label, "format": str(image_format)},
                )
            rotated = int(image.getexif().get(_ORIENTATION_TAG, 1)) not in (0, 1)
            upright = ImageOps.exif_transpose(image)
            width, height = upright.size
            if width < 1 or height < 1:
                raise IngestionError(f"image has no pixels: {label}", {"file": label})
            if not rotated and upright.mode in _NATIVE_MODES:
                return data, (width, height)
            target = upright if upright.mode in _NATIVE_MODES else upright.convert("RGB")
            buffer = io.BytesIO()
            target.save(buffer, format="PNG")
            return buffer.getvalue(), (width, height)
    except IngestionError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise IngestionError(
            f"cannot read image {label}: {exc}", {"file": label, "error": str(exc)}
        ) from exc


def images_to_pdf_bytes(images: list[bytes]) -> bytes:
    """Wrap images as a PDF with one page per image.

    Parameters
    ----------
    images : list of bytes
        Raw PNG or JPEG file contents, in page order.

    Returns
    -------
    bytes
        PDF binary whose page ``N`` holds image ``N`` and no text layer.

    Raises
    ------
    IngestionError
        If the list is empty or any image is corrupt or unsupported.

    Notes
    -----
    Each page is sized to the image pixel dimensions interpreted at 200 dpi
    (``pixels * 72 / 200`` points), the same resolution the inspector uses to
    render pages for OCR. Rendering a wrapped page therefore reproduces the
    original pixel grid and OCR bounding boxes, expressed in points, map back
    to image pixels by a constant factor. The image fills the whole page. The
    output is serialised deterministically (no object-stream or date
    variation) for identical inputs.
    """
    if not images:
        raise IngestionError("no images provided")
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for index, data in enumerate(images, start=1):
            embedded, (width, height) = _prepare_image(data, f"image {index}")
            page_width = width * _POINTS_PER_INCH / RENDER_DPI
            page_height = height * _POINTS_PER_INCH / RENDER_DPI
            page = document.new_page(width=page_width, height=page_height)
            page.insert_image(page.rect, stream=embedded)  # type: ignore[no-untyped-call]
        return bytes(document.tobytes(deflate=True, garbage=4, no_new_id=True))  # type: ignore[no-untyped-call]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            "cannot wrap images as PDF", {"error": str(exc), "image_count": len(images)}
        ) from exc
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _hash_input(blobs: list[bytes]) -> bytes:
    """Build the idempotency hash input for a list of image files.

    Parameters
    ----------
    blobs : list of bytes
        Raw image bytes in page order.

    Returns
    -------
    bytes
        The raw bytes for one image; otherwise each image preceded by its
        8-byte big-endian length, in order, so different splits of the same
        byte stream never collide.
    """
    if len(blobs) == 1:
        return blobs[0]
    return b"".join(_LENGTH_PREFIX.pack(len(blob)) + blob for blob in blobs)


def load_document_bytes(paths: list[Path]) -> tuple[bytes, bytes]:
    """Read image files and prepare them for the pipeline.

    Parameters
    ----------
    paths : list of Path
        Image files forming one document, in page order.

    Returns
    -------
    tuple of (bytes, bytes)
        ``(original_bytes_for_hash, pdf_bytes_for_processing)``. The first
        element is the file content itself for a single image, or the
        length-prefixed concatenation of all files for several images; the
        second is the wrapped PDF.

    Raises
    ------
    IngestionError
        If no path is given, a file is unreadable, empty, has an
        unsupported suffix, or is not a valid image.
    """
    if not paths:
        raise IngestionError("no image files provided")
    blobs: list[bytes] = []
    for path in paths:
        if not is_image_path(path):
            raise IngestionError(
                f"unsupported image suffix: {path.name}",
                {"path": str(path), "supported": list(SUPPORTED_IMAGE_SUFFIXES)},
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise IngestionError(
                f"cannot read document: {path}", {"path": str(path), "error": str(exc)}
            ) from exc
        if not data:
            raise IngestionError(f"document is empty: {path}", {"path": str(path)})
        blobs.append(data)
    return _hash_input(blobs), images_to_pdf_bytes(blobs)


__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "images_to_pdf_bytes",
    "is_image_path",
    "load_document_bytes",
]
