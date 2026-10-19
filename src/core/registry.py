"""Schema-driven document type registry (Open/Closed principle).

The pipeline orchestrator looks up a :class:`DocumentTypeSpec` by its
identifier instead of hardcoding field lists, so new document types are
added by registering a spec rather than modifying pipeline code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Optional

from src.core.exceptions import UnknownDocumentTypeError

PromptBuilder = Callable[[str], str]
"""Callable receiving the XML-wrapped ``<document_pages>`` block and returning the user prompt."""

ACTA_CONSTITUTIVA: str = "acta_constitutiva"
"""Identifier of the Mexican incorporation deed document type."""

ACTA_CONSTITUTIVA_FIELDS: tuple[str, ...] = (
    "razon_social",
    "fecha_constitucion",
    "numero_escritura",
    "notario",
    "socios",
    "representantes_legales",
    "poderes",
    "capital_social_actual",
)
"""The 8 required serialized fields of an Acta Constitutiva, in contract order."""

ACTA_CONSTITUTIVA_CRITICAL_FIELDS: tuple[str, ...] = (
    "fecha_constitucion",
    "numero_escritura",
    "notario",
    "representantes_legales",
    "poderes",
    "capital_social_actual",
)
"""Fields whose misreading on a scanned page carries legal or financial risk."""


@dataclass(frozen=True)
class DocumentTypeSpec:
    """Declarative description of a supported document type.

    Parameters
    ----------
    document_type : str
        Unique identifier (for example ``"acta_constitutiva"``).
    display_name : str
        Human-readable name used in logs and reports.
    required_fields : tuple of str
        Serialized field names that must be present; missing ones are
        reported with status ``no_encontrado``.
    applicable_fields : tuple of str, optional
        Fields that legally apply to the type. Defaults to
        ``required_fields``. Fields outside this set may be reported as
        ``no_aplicable``.
    temporal_resolution : bool, default True
        Whether the temporal resolver must run on the amendment history.
    prompt_builder : PromptBuilder or None, optional
        Hook producing the type-specific user prompt from the page block.
        ``None`` lets the LLM provider use its default prompt.
    description : str, default ""
        Free-text description.
    critical_fields : tuple of str, optional
        Applicable fields that, on a scanned page, always go through the
        VLM re-read of their evidence region and the Decisions predicate on
        the same crop (independently of OCR confidence), and whose absence
        on a scanned document triggers one re-read and re-extraction.

    Raises
    ------
    ValueError
        If ``document_type`` is empty, ``required_fields`` is empty or
        contains duplicates, or a required or critical field is not
        applicable.
    """

    document_type: str
    display_name: str
    required_fields: tuple[str, ...]
    applicable_fields: tuple[str, ...] = field(default_factory=tuple)
    temporal_resolution: bool = True
    prompt_builder: Optional[PromptBuilder] = None
    description: str = ""
    critical_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the spec and default ``applicable_fields``.

        Raises
        ------
        ValueError
            If the spec is internally inconsistent.
        """
        if not self.document_type.strip():
            raise ValueError("document_type must be a non-empty string")
        if not self.required_fields:
            raise ValueError("required_fields must not be empty")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("required_fields must not contain duplicates")
        if not self.applicable_fields:
            object.__setattr__(self, "applicable_fields", tuple(self.required_fields))
        missing = set(self.required_fields) - set(self.applicable_fields)
        if missing:
            raise ValueError(f"required fields not applicable: {sorted(missing)}")
        not_applicable = set(self.critical_fields) - set(self.applicable_fields)
        if not_applicable:
            raise ValueError(f"critical fields not applicable: {sorted(not_applicable)}")

    def is_critical(self, field_name: str) -> bool:
        """Return whether a field is critical for this document type.

        Parameters
        ----------
        field_name : str
            Serialized field name.

        Returns
        -------
        bool
            ``True`` if the field is in ``critical_fields``.
        """
        return field_name in self.critical_fields

    def is_required(self, field_name: str) -> bool:
        """Return whether a field is required for this document type.

        Parameters
        ----------
        field_name : str
            Serialized field name.

        Returns
        -------
        bool
            ``True`` if the field is in ``required_fields``.
        """
        return field_name in self.required_fields

    def is_applicable(self, field_name: str) -> bool:
        """Return whether a field legally applies to this document type.

        Parameters
        ----------
        field_name : str
            Serialized field name.

        Returns
        -------
        bool
            ``True`` if the field is in ``applicable_fields``.
        """
        return field_name in self.applicable_fields

    def build_prompt(self, pages_block: str) -> Optional[str]:
        """Build the type-specific user prompt, if a builder is registered.

        Parameters
        ----------
        pages_block : str
            The ``<document_pages>...</document_pages>`` XML block.

        Returns
        -------
        str or None
            The prompt, or ``None`` when no builder is registered.
        """
        if self.prompt_builder is None:
            return None
        return self.prompt_builder(pages_block)


class DocumentRegistry:
    """Registry mapping document type identifiers to specs.

    Parameters
    ----------
    specs : iterable of DocumentTypeSpec, optional
        Initial specs. When ``None``, the default ``acta_constitutiva``
        spec is pre-registered.

    Notes
    -----
    Use :meth:`DocumentRegistry.default` for the shared process-wide
    instance; create new instances in tests for isolation.
    """

    _default_instance: Optional["DocumentRegistry"] = None

    def __init__(self, specs: Optional[list[DocumentTypeSpec]] = None) -> None:
        """Initialise the registry.

        Parameters
        ----------
        specs : list of DocumentTypeSpec, optional
            Initial specs; defaults to the built-in Acta Constitutiva spec.
        """
        self._specs: dict[str, DocumentTypeSpec] = {}
        for spec in specs if specs is not None else [build_acta_constitutiva_spec()]:
            self.register(spec)

    @classmethod
    def default(cls) -> "DocumentRegistry":
        """Return the shared process-wide registry.

        Returns
        -------
        DocumentRegistry
            Lazily created registry with built-in types pre-registered.
        """
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def register(self, spec: DocumentTypeSpec, *, replace: bool = False) -> None:
        """Register a document type spec.

        Parameters
        ----------
        spec : DocumentTypeSpec
            Spec to register.
        replace : bool, default False
            Allow overwriting an existing spec with the same identifier.

        Raises
        ------
        ValueError
            If the identifier is already registered and ``replace`` is False.
        """
        if spec.document_type in self._specs and not replace:
            raise ValueError(f"document type already registered: {spec.document_type}")
        self._specs[spec.document_type] = spec

    def get(self, document_type: str) -> DocumentTypeSpec:
        """Look up a spec by identifier.

        Parameters
        ----------
        document_type : str
            Registered identifier.

        Returns
        -------
        DocumentTypeSpec
            The registered spec.

        Raises
        ------
        UnknownDocumentTypeError
            If the identifier is not registered.
        """
        try:
            return self._specs[document_type]
        except KeyError as error:
            raise UnknownDocumentTypeError(
                f"unknown document type: {document_type}",
                details={"registered": sorted(self._specs)},
            ) from error

    def is_registered(self, document_type: str) -> bool:
        """Return whether an identifier is registered.

        Parameters
        ----------
        document_type : str
            Identifier to check.

        Returns
        -------
        bool
            ``True`` if registered.
        """
        return document_type in self._specs

    def document_types(self) -> list[str]:
        """Return all registered identifiers, sorted.

        Returns
        -------
        list of str
            Registered identifiers.
        """
        return sorted(self._specs)

    def __contains__(self, document_type: object) -> bool:
        """Support ``"acta_constitutiva" in registry``.

        Parameters
        ----------
        document_type : object
            Candidate identifier.

        Returns
        -------
        bool
            ``True`` if registered.
        """
        return isinstance(document_type, str) and document_type in self._specs

    def __iter__(self) -> Iterator[DocumentTypeSpec]:
        """Iterate over registered specs in identifier order.

        Returns
        -------
        Iterator of DocumentTypeSpec
            Registered specs.
        """
        return iter([self._specs[key] for key in sorted(self._specs)])

    def __len__(self) -> int:
        """Return the number of registered document types.

        Returns
        -------
        int
            Count of specs.
        """
        return len(self._specs)


def build_acta_constitutiva_spec(prompt_builder: Optional[PromptBuilder] = None) -> DocumentTypeSpec:
    """Build the built-in Acta Constitutiva spec.

    Parameters
    ----------
    prompt_builder : PromptBuilder or None, optional
        Optional prompt hook; the LLM provider may register one later via
        ``DocumentRegistry.register(..., replace=True)``.

    Returns
    -------
    DocumentTypeSpec
        Spec with the 8 required fields and temporal resolution enabled.
    """
    return DocumentTypeSpec(
        document_type=ACTA_CONSTITUTIVA,
        display_name="Acta Constitutiva",
        required_fields=ACTA_CONSTITUTIVA_FIELDS,
        temporal_resolution=True,
        prompt_builder=prompt_builder,
        description="Mexican incorporation deed and subsequent protocolized amendments.",
        critical_fields=ACTA_CONSTITUTIVA_CRITICAL_FIELDS,
    )
