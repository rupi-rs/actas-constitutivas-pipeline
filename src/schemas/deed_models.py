"""Pydantic v2 data contracts for Mexican corporate deed extraction.

Defines the evidence model, the generic :class:`FieldWithEvidence` wrapper,
the domain entity sub-models, the pre-resolution :class:`RawDeedExtraction`
returned by LLM providers (and stored in the SHA-256 cache) and the final
:class:`CorporateDeedOutput` contract.

Notes
-----
Serialized field names are the Spanish keys required by the evaluation
spec (``razon_social``, ``fecha_constitucion``, ...). Because the Python
attribute names already equal those keys, no explicit aliases are
declared; ``model_dump(by_alias=True, mode="json")`` and
``model_validate`` therefore round-trip losslessly. All models forbid
unknown keys (``extra="forbid"``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: str = "1.0.0"
"""Version of the output contract recorded in :class:`AuditMetadata`."""

T = TypeVar("T")


class StrictModel(BaseModel):
    """Base model applying the shared strict configuration.

    Notes
    -----
    ``populate_by_name=True`` and ``extra="forbid"`` per spec 01.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class FieldStatus(str, Enum):
    """Extraction status of a field.

    Notes
    -----
    ``EXTRACTED`` when present and legible, ``NOT_FOUND`` when required but
    missing or illegible, ``NOT_APPLICABLE`` when legally inapplicable.
    """

    EXTRACTED = "extraido"
    NOT_FOUND = "no_encontrado"
    NOT_APPLICABLE = "no_aplicable"


class ConfidenceSource(str, Enum):
    """Origin of an evidence confidence score."""

    NATIVE_PDF = "native_pdf"
    OCR_ENGINE = "ocr_engine"
    DECISIONS_PREDICATE = "decisions_predicate"
    LLM_LOGPROBS = "llm_logprobs"
    DETERMINISTIC_RULE = "deterministic_rule"


class AmendmentType(str, Enum):
    """Type of chronological corporate act."""

    INCORPORATION = "constitucion_original"
    CAPITAL_INCREASE = "aumento_capital"
    GRANT_OF_POWERS = "otorgamiento_poderes"
    REVOCATION = "revocacion_poderes"


class BoundingBox(StrictModel):
    """Axis-aligned rectangle in PDF points (origin top-left).

    Parameters
    ----------
    x1, y1 : float
        Top-left corner.
    x2, y2 : float
        Bottom-right corner.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, coordinates: list[float] | tuple[float, ...]) -> "BoundingBox":
        """Build a box from a ``[x1, y1, x2, y2]`` sequence.

        Parameters
        ----------
        coordinates : list or tuple of float
            Exactly four coordinates.

        Returns
        -------
        BoundingBox
            The constructed box.

        Raises
        ------
        ValueError
            If the sequence does not have exactly four items.
        """
        if len(coordinates) != 4:
            raise ValueError("bounding box requires exactly four coordinates")
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    def as_list(self) -> list[float]:
        """Return the box as ``[x1, y1, x2, y2]``.

        Returns
        -------
        list of float
            Coordinates in contract order.
        """
        return [self.x1, self.y1, self.x2, self.y2]

    def overlaps(self, other: "BoundingBox") -> bool:
        """Return whether this box intersects another with positive area.

        Parameters
        ----------
        other : BoundingBox
            Box to compare against.

        Returns
        -------
        bool
            ``True`` if the intersection has positive width and height.
        """
        return min(self.x2, other.x2) > max(self.x1, other.x1) and min(self.y2, other.y2) > max(
            self.y1, other.y1
        )


class Evidence(StrictModel):
    """Grounding evidence for an extracted value.

    Parameters
    ----------
    page_number : int
        1-indexed page where the information was found.
    source_quote : str
        Verbatim substring of that page.
    confidence_score : float
        Confidence in ``[0.0, 1.0]``.
    confidence_source : ConfidenceSource
        Origin of the confidence score.
    grounding_verified : bool, default False
        Set by the deterministic grounding check.
    bounding_box : BoundingBox or None, default None
        Location of the quote on the page, when known.
    """

    page_number: int = Field(ge=1)
    source_quote: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_source: ConfidenceSource
    grounding_verified: bool = False
    bounding_box: Optional[BoundingBox] = None


class FieldWithEvidence(StrictModel, Generic[T]):
    """Generic wrapper pairing an extracted value with status and evidence.

    Parameters
    ----------
    value : T or None, default None
        Extracted value; parameterize as ``FieldWithEvidence[str]``,
        ``FieldWithEvidence[NotaryInfo]``,
        ``FieldWithEvidence[list[ShareholderItem]]``, etc.
    status : FieldStatus
        Extraction status.
    evidence : Evidence or None, default None
        Primary grounding evidence.
    requires_human_review : bool, default False
        Whether the field must be reviewed by a human.
    review_reason : str or None, default None
        Reason(s) for review; multiple reasons are joined by ``"; "``.
    """

    value: Optional[T] = None
    status: FieldStatus
    evidence: Optional[Evidence] = None
    requires_human_review: bool = False
    review_reason: Optional[str] = None

    def flag_for_review(self, reason: str) -> None:
        """Mark the field for human review, appending a reason.

        Parameters
        ----------
        reason : str
            Reason to append (deduplicated, joined with ``"; "``).
        """
        self.requires_human_review = True
        if not self.review_reason:
            self.review_reason = reason
        elif reason not in self.review_reason.split("; "):
            self.review_reason = f"{self.review_reason}; {reason}"


class NotaryInfo(StrictModel):
    """Notary public who authenticated the deed.

    Parameters
    ----------
    nombre : str
        Notary full name.
    numero_notaria : int
        Notary office number.
    entidad_federativa : str
        Mexican federal entity of the notary office.
    """

    nombre: str
    numero_notaria: int
    entidad_federativa: str


class ShareholderItem(StrictModel):
    """Shareholder (socio) entry.

    Parameters
    ----------
    nombre : str
        Shareholder name as written.
    rfc : str or None, default None
        Mexican tax identifier, if stated.
    acciones : int
        Number of shares or social parts.
    valor_aportacion : float
        Contribution amount in MXN.
    porcentaje : float
        Ownership percentage (0-100).
    evidence : Evidence
        Grounding evidence.
    """

    nombre: str
    rfc: Optional[str] = None
    acciones: int
    valor_aportacion: float
    porcentaje: float
    evidence: Evidence


class LegalRepresentativeItem(StrictModel):
    """Legal representative entry.

    Parameters
    ----------
    nombre : str
        Name as written in the deed.
    nombre_normalizado : str
        Uppercase, accent-free, title-free name.
    cargo : str
        Position (for example ``"Administrador Unico"``).
    vigente : bool, default True
        ``False`` once revoked.
    fecha_nombramiento : str
        Appointment date, ``YYYY-MM-DD``.
    evidence : Evidence
        Grounding evidence.
    """

    nombre: str
    nombre_normalizado: str
    cargo: str
    vigente: bool = True
    fecha_nombramiento: str
    evidence: Evidence


class PowerOfAttorneyItem(StrictModel):
    """Power of attorney grant (one item per power type and person).

    Parameters
    ----------
    otorgado_a : str
        Grantee name.
    tipo_poder : str
        Power type, for example ``"Poder General para Pleitos y Cobranzas"``.
    modalidad : str
        ``"Individual"`` or ``"Mancomunado"``.
    vigente : bool, default True
        ``False`` once revoked.
    evidence : Evidence
        Grounding evidence.
    """

    otorgado_a: str
    tipo_poder: str
    modalidad: str
    vigente: bool = True
    evidence: Evidence


class CapitalInfo(StrictModel):
    """Current share capital.

    Parameters
    ----------
    monto_numerico : float
        Amount as a number.
    moneda : str, default "MXN"
        Currency code.
    monto_en_letra : str
        Amount in Spanish words.
    fecha_acto_vigente : str
        Date of the deed establishing this amount, ``YYYY-MM-DD``.
    escritura_origen : str
        Deed number establishing this amount.
    """

    monto_numerico: float
    moneda: str = "MXN"
    monto_en_letra: str
    fecha_acto_vigente: str
    escritura_origen: str


class AmendmentEvent(StrictModel):
    """Chronological corporate act used by the temporal resolver.

    Parameters
    ----------
    event_type : AmendmentType
        Kind of act.
    fecha_acto : str
        Notarized act date, ``YYYY-MM-DD``.
    numero_escritura : str
        Deed number of the act.
    descripcion : str
        Short description.
    sujeto_afectado : str or None, default None
        Person affected (appointee or revoked individual).
    monto_nuevo : float or None, default None
        New capital amount for capital modifications.
    matched_canonical_entity : str or None, default None
        Canonical normalized name matched by the resolver.
    fuzzy_match_score : float or None, default None
        Resolver similarity score (0-100).
    evidence : Evidence
        Grounding evidence.
    revoked_appointment_evidence : list of Evidence, default []
        Schema extension beyond spec 01, filled by the resolver only for
        ``REVOCATION`` events: a deep copy of the appointment evidence of
        every representative revoked by the act, so the output retains
        both the appointment and the revocation evidence in structured
        form. The empty default keeps cached extractions and ground truth
        files (which omit the key) valid.
    """

    event_type: AmendmentType
    fecha_acto: str
    numero_escritura: str
    descripcion: str
    sujeto_afectado: Optional[str] = None
    monto_nuevo: Optional[float] = None
    matched_canonical_entity: Optional[str] = None
    fuzzy_match_score: Optional[float] = None
    evidence: Evidence
    revoked_appointment_evidence: list[Evidence] = Field(default_factory=list)


class AuditMetadata(StrictModel):
    """Processing metadata attached to every output.

    Parameters
    ----------
    job_id : str
        Unique job identifier.
    doc_sha256 : str
        SHA-256 of the source PDF.
    document_filename : str
        Source file name.
    schema_version : str, default "1.0.0"
        Output contract version.
    model_version : str
        Extraction model or cache version.
    total_pages : int
        Page count.
    native_pages_count : int
        Pages extracted natively.
    ocr_pages_count : int
        Pages routed to OCR.
    global_human_review_required : bool
        ``True`` if any field requires review.
    processing_time_ms : float
        Wall-clock processing time.
    """

    job_id: str
    doc_sha256: str
    document_filename: str
    schema_version: str = SCHEMA_VERSION
    model_version: str
    total_pages: int
    native_pages_count: int
    ocr_pages_count: int
    global_human_review_required: bool
    processing_time_ms: float


class _DeedFields(StrictModel):
    """Shared 8 contract fields plus the amendment history.

    Notes
    -----
    Internal base; use :class:`RawDeedExtraction` or
    :class:`CorporateDeedOutput`.
    """

    razon_social: FieldWithEvidence[str]
    fecha_constitucion: FieldWithEvidence[str]
    numero_escritura: FieldWithEvidence[str]
    notario: FieldWithEvidence[NotaryInfo]
    socios: FieldWithEvidence[list[ShareholderItem]]
    representantes_legales: FieldWithEvidence[list[LegalRepresentativeItem]]
    poderes: FieldWithEvidence[list[PowerOfAttorneyItem]]
    capital_social_actual: FieldWithEvidence[CapitalInfo]
    amendment_history: list[AmendmentEvent] = Field(default_factory=list)


class RawDeedExtraction(_DeedFields):
    """Pre-resolution extraction returned by LLM providers and cached.

    Notes
    -----
    Person and power lists contain everyone ever appointed; the temporal
    resolver subtracts revoked individuals. No ``audit_metadata``.
    """


class CorporateDeedOutput(_DeedFields):
    """Final resolved, validated output contract.

    Notes
    -----
    Serialize with ``model_dump(by_alias=True, mode="json")``.
    """

    audit_metadata: AuditMetadata


def raw_extraction_json_schema() -> dict[str, Any]:
    """Return the JSON schema of :class:`RawDeedExtraction`.

    Returns
    -------
    dict of str to Any
        Pydantic JSON schema (serialization by alias).

    Notes
    -----
    This plain Pydantic schema is not OpenAI strict-mode compatible as-is
    (fields with defaults are optional and ``default`` keywords are
    emitted). For Structured Outputs use :func:`openai_strict_json_schema`
    instead, which marks every property required (nullable where
    optional), sets ``additionalProperties: false`` and strips ``default``
    keywords; the LLM provider sends that schema as a strict
    ``json_schema`` response format and validates the reply with
    ``RawDeedExtraction.model_validate_json``.
    """
    return RawDeedExtraction.model_json_schema(by_alias=True)


def _strip_defaults(node: Any) -> Any:
    """Recursively remove ``default`` keywords from a JSON schema.

    Parameters
    ----------
    node : Any
        Schema node (dict, list or scalar).

    Returns
    -------
    Any
        Copy of the node without ``default`` keys.
    """
    if isinstance(node, dict):
        return {key: _strip_defaults(item) for key, item in node.items() if key != "default"}
    if isinstance(node, list):
        return [_strip_defaults(item) for item in node]
    return node


def openai_strict_json_schema(model: type[BaseModel] = RawDeedExtraction) -> dict[str, Any]:
    """Return an OpenAI Structured Outputs strict-mode JSON schema.

    Parameters
    ----------
    model : type of BaseModel, default RawDeedExtraction
        Model to convert.

    Returns
    -------
    dict of str to Any
        Schema with every property required (optional ones nullable),
        ``additionalProperties: false`` and no ``default`` keywords.

    Notes
    -----
    Delegates to the OpenAI SDK strict converter (imported lazily) and then
    strips ``default`` keywords, which strict mode does not need. Numeric
    ``minimum`` / ``maximum`` bounds are retained.
    """
    from openai.lib._pydantic import to_strict_json_schema

    schema: dict[str, Any] = _strip_defaults(to_strict_json_schema(model))
    return schema
