"""File-system persistence for the SHA-256 cache and final outputs.

All writes are atomic: content is written to a temporary file in the target
directory and moved into place with ``os.replace``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.core.config import Settings
from src.core.exceptions import EvaluationError, StorageError
from src.providers.interfaces import BaseStorageRepository
from src.schemas.deed_models import CorporateDeedOutput

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def save_json(path: Path, payload: Any) -> Path:
    """Atomically write a JSON document.

    Parameters
    ----------
    path : Path
        Destination file; parent directories are created.
    payload : Any
        JSON-serialisable object.

    Returns
    -------
    Path
        The written path.

    Raises
    ------
    StorageError
        If serialisation or writing fails.

    Notes
    -----
    Encoding is UTF-8 with ``ensure_ascii=False`` and ``indent=2``.
    """
    target = Path(path)
    tmp_name: Optional[str] = None
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        tmp_name = None
    except (OSError, TypeError, ValueError) as exc:
        raise StorageError(
            f"failed to write JSON file: {target}", {"path": str(target), "error": str(exc)}
        ) from exc
    finally:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
    return target


def _read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    Any
        Parsed content.

    Raises
    ------
    StorageError
        If the file is unreadable or contains invalid JSON.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StorageError(
            f"failed to read JSON file: {path}", {"path": str(path), "error": str(exc)}
        ) from exc


def _validate_identifier(value: str, label: str) -> None:
    """Reject identifiers that could escape the target directory.

    Parameters
    ----------
    value : str
        Identifier to check.
    label : str
        Name used in the error message.

    Raises
    ------
    StorageError
        If the identifier contains unsafe characters.
    """
    if not value or not _SAFE_ID.match(value) or value in {".", ".."}:
        raise StorageError(f"invalid {label}: {value!r}", {label: value})


class FileSystemRepository(BaseStorageRepository):
    """Local file-system implementation of the storage abstraction.

    Parameters
    ----------
    settings : Settings
        Pipeline settings providing cache, outputs and ground-truth paths.
    """

    def __init__(self, settings: Settings) -> None:
        """Store the settings.

        Parameters
        ----------
        settings : Settings
            Pipeline settings.
        """
        self._settings = settings

    def load_cache(self, doc_sha256: str) -> Optional[dict[str, Any]]:
        """Load a cache entry.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.

        Returns
        -------
        dict of str to Any or None
            Cache payload, or ``None`` on a miss.

        Raises
        ------
        StorageError
            If the hash is malformed or the file is corrupt.
        """
        _validate_identifier(doc_sha256, "doc_sha256")
        path = self._settings.cache_file_for(doc_sha256)
        if not path.is_file():
            return None
        data = _read_json(path)
        if not isinstance(data, dict):
            raise StorageError(f"cache entry is not a JSON object: {path}", {"path": str(path)})
        return data

    def save_cache(self, doc_sha256: str, payload: dict[str, Any]) -> Path:
        """Persist a cache entry atomically.

        Parameters
        ----------
        doc_sha256 : str
            Document hash.
        payload : dict of str to Any
            Cache payload.

        Returns
        -------
        Path
            Written file path.

        Raises
        ------
        StorageError
            If the hash is malformed or writing fails.
        """
        _validate_identifier(doc_sha256, "doc_sha256")
        return save_json(self._settings.cache_file_for(doc_sha256), payload)

    def save_output(self, acta_id: str, output: CorporateDeedOutput) -> Path:
        """Persist a final output as ``outputs/{acta_id}.json``.

        Parameters
        ----------
        acta_id : str
            Output identifier.
        output : CorporateDeedOutput
            Final output.

        Returns
        -------
        Path
            Written file path.

        Raises
        ------
        StorageError
            If the identifier is unsafe or writing fails.

        Notes
        -----
        Serialised with ``by_alias=True`` so Spanish contract aliases are kept.
        """
        _validate_identifier(acta_id, "acta_id")
        payload = output.model_dump(mode="json", by_alias=True)
        return save_json(self._settings.outputs_dir / f"{acta_id}.json", payload)

    def load_output(self, acta_id: str) -> Optional[dict[str, Any]]:
        """Load a final output as a JSON dictionary.

        Parameters
        ----------
        acta_id : str
            Output identifier.

        Returns
        -------
        dict of str to Any or None
            Output payload, or ``None`` if absent.

        Raises
        ------
        StorageError
            If the identifier is unsafe or the file is corrupt.
        """
        _validate_identifier(acta_id, "acta_id")
        path = self._settings.outputs_dir / f"{acta_id}.json"
        if not path.is_file():
            return None
        data = _read_json(path)
        if not isinstance(data, dict):
            raise StorageError(f"output is not a JSON object: {path}", {"path": str(path)})
        return data

    def load_ground_truth(self) -> dict[str, Any]:
        """Load the evaluation ground truth.

        Returns
        -------
        dict of str to Any
            Ground truth keyed by acta identifier.

        Raises
        ------
        EvaluationError
            If the ground truth file is missing.
        StorageError
            If the file is corrupt.
        """
        path = self._settings.ground_truth_path
        if not path.is_file():
            raise EvaluationError(
                f"ground truth file not found: {path}", {"path": str(path)}
            )
        data = _read_json(path)
        if not isinstance(data, dict):
            raise StorageError(f"ground truth is not a JSON object: {path}", {"path": str(path)})
        return data
