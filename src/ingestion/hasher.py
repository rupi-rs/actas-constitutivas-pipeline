"""SHA-256 hashing of document binaries.

The digest is the idempotency key for the extraction cache
(``data/cache/{sha256}.json``) and is attached to every audit record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.exceptions import IngestionError

_CHUNK_SIZE: int = 1024 * 1024


def compute_sha256(data: bytes) -> str:
    """Compute the SHA-256 digest of a binary payload.

    Parameters
    ----------
    data : bytes
        Raw document bytes.

    Returns
    -------
    str
        Lowercase hexadecimal digest (64 characters).

    Raises
    ------
    IngestionError
        If ``data`` is empty.

    Notes
    -----
    An empty payload is rejected because it can never be a valid PDF and
    would otherwise map every empty upload to the same cache key.
    """
    if not data:
        raise IngestionError("cannot hash an empty document payload")
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file, streaming in chunks.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest (64 characters).

    Raises
    ------
    IngestionError
        If the file does not exist, cannot be read, or is empty.

    Notes
    -----
    The file is read in 1 MiB chunks so large scans do not need to fit in
    memory for hashing.
    """
    digest = hashlib.sha256()
    total = 0
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise IngestionError(
            f"cannot read document for hashing: {path}", {"path": str(path), "error": str(exc)}
        ) from exc
    if total == 0:
        raise IngestionError(f"document is empty: {path}", {"path": str(path)})
    return digest.hexdigest()
