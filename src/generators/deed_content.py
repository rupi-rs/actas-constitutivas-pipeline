"""Authoritative content for the three synthetic incorporation deeds.

This module holds data only: the page text of ``acta_a.pdf``,
``acta_b.pdf`` and ``acta_c.pdf`` (spec 02), the pre-verified raw
extractions stored in the SHA-256 cache (Appendix B of the architecture
contract) and the ground truth expectations used by the evaluator. Layout,
rendering and file IO live in :mod:`src.generators.synthetic_deeds`.

Notes
-----
Every clause that spec 02 mandates verbatim is declared once as a module
constant and reused both in the page layout and in the ``source_quote``
strings, so the PDF text and the cached evidence cannot drift apart.

Page geometry is US Letter (612 x 792 PDF points). Raster region bounding
boxes are expressed in PDF points with the origin at the top-left corner of
the page, matching the coordinates reported by PyMuPDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Callable
from typing import Any, Optional, Union

from src.providers.interfaces import OCRRegion, OCRResult
from src.schemas.deed_models import (
    AmendmentEvent,
    AmendmentType,
    BoundingBox,
    CapitalInfo,
    ConfidenceSource,
    Evidence,
    FieldStatus,
    FieldWithEvidence,
    LegalRepresentativeItem,
    NotaryInfo,
    PowerOfAttorneyItem,
    RawDeedExtraction,
    ShareholderItem,
)

CACHE_MODEL_VERSION: str = "pre-verified-cache-v1"
"""Model version recorded in every pre-verified cache payload."""

DOCUMENT_TYPE: str = "acta_constitutiva"
"""Registered document type of the synthetic deeds."""

OCR_ENGINE_NAME: str = "deterministic-ocr-v1"
"""Engine identifier recorded in the cached OCR result."""

POWER_LITIGATION: str = "Poder General para Pleitos y Cobranzas"
"""Canonical power type: litigation and collections."""

POWER_ADMINISTRATION: str = "Poder General para Actos de Administración"
"""Canonical power type: acts of administration."""

POWER_OWNERSHIP: str = "Poder General para Actos de Dominio"
"""Canonical power type: acts of ownership."""

MODALITY_INDIVIDUAL: str = "Individual"
"""Power exercise modality used by every synthetic grant."""

ACTA_B_CAPITAL_BBOX: tuple[float, float, float, float] = (72.0, 410.0, 520.0, 455.0)
"""Bounding box (PDF points, top-left origin) of the sealed capital clause."""

ACTA_B_CAPITAL_OCR_CONFIDENCE: float = 0.72
"""OCR confidence of the sealed capital clause region (Escalation Tier 1)."""

ACTA_B_VERIFIER_PROBABILITY: float = 0.78
"""Cached Decisions predicate probability (Escalation Tier 2)."""

ACTA_B_VERIFIER_PREDICATE: str = "The capital social amount on page 2 equals $250,000.00 MXN"
"""Predicate evaluated by the verifier for the Acta B capital clause."""

ACTA_B_SEAL_TEXT: str = "[SELLO NOTARIAL ILEGIBLE - COPIA CERTIFICADA]"
"""Text printed inside the simulated notarial seal on Acta B page 2."""

ACTA_B_NOISE_SEED: int = 20200722
"""Seed of the deterministic scan noise on Acta B page 2."""


class BlockStyle(str, Enum):
    """Typographic role of a native text block."""

    TITLE = "title"
    HEADING = "heading"
    BODY = "body"
    SIGNATURE = "signature"


@dataclass(frozen=True)
class TextBlock:
    """A paragraph rendered as native PDF text.

    Parameters
    ----------
    text : str
        Paragraph text; wrapped only at spaces, never hyphenated.
    style : BlockStyle, default BlockStyle.BODY
        Typographic role.
    mandated : bool, default False
        ``True`` when spec 02 requires this exact text on the page.
    """

    text: str
    style: BlockStyle = BlockStyle.BODY
    mandated: bool = False


@dataclass(frozen=True)
class NativePageSpec:
    """A digitally born page with extractable text.

    Parameters
    ----------
    blocks : tuple of TextBlock
        Paragraphs in reading order.
    running_header : str
        Small header printed at the top of the page.
    footer : str
        Small footer printed at the bottom of the page.
    """

    blocks: tuple[TextBlock, ...]
    running_header: str
    footer: str


@dataclass(frozen=True)
class RasterRegionSpec:
    """A text region drawn onto a scanned page image.

    Parameters
    ----------
    text : str
        Region text as printed (and as recovered by OCR).
    bbox : tuple of float
        ``(x1, y1, x2, y2)`` in PDF points, top-left origin. The renderer
        wraps the text inside this box and fails if it does not fit.
    ocr_confidence : float
        Confidence reported by the OCR engine for this region.
    mandated : bool, default False
        ``True`` when spec 02 requires this exact text on the page.
    """

    text: str
    bbox: tuple[float, float, float, float]
    ocr_confidence: float
    mandated: bool = False


@dataclass(frozen=True)
class RasterPageSpec:
    """A scanned page rendered as an image with no native text.

    Parameters
    ----------
    regions : tuple of RasterRegionSpec
        Text regions in reading order.
    seal_text : str
        Text printed inside the semi-transparent notarial seal.
    seal_center_bbox : tuple of float
        Region (PDF points) the seal is centred on.
    noise_seed : int
        Seed for deterministic speckle noise.
    """

    regions: tuple[RasterRegionSpec, ...]
    seal_text: str
    seal_center_bbox: tuple[float, float, float, float]
    noise_seed: int


PageSpec = Union[NativePageSpec, RasterPageSpec]
"""Either a native or a raster page specification."""


@dataclass(frozen=True)
class DeedSpec:
    """Complete specification of one synthetic deed.

    Parameters
    ----------
    acta_id : str
        Identifier (``"acta_a"``, ``"acta_b"``, ``"acta_c"``).
    filename : str
        PDF file name inside ``data/samples``.
    title : str
        PDF metadata title.
    scenario : str
        Short English description of the test scenario.
    pages : tuple of PageSpec
        Pages in order (1-indexed when referenced by evidence).
    """

    acta_id: str
    filename: str
    title: str
    scenario: str
    pages: tuple[PageSpec, ...] = field(default_factory=tuple)

    @property
    def total_pages(self) -> int:
        """int: Number of pages in the deed."""
        return len(self.pages)


# ---------------------------------------------------------------------------
# Verbatim clauses mandated by spec 02
# ---------------------------------------------------------------------------

A_P1_HEADER = "INSTRUMENTO PUBLICO NUMERO 45,120 (CUARENTA Y CINCO MIL CIENTO VEINTE)."
A_P1_DATE_PLACE = (
    "En la Ciudad de Mexico, a los 15 dias del mes de marzo de 2021, yo, el Licenciado "
    "ROBERTO MENDEZ LARA, Titular de la Notaria Publica Numero 88 de la Ciudad de Mexico, "
    "hago constar el CONTRATO DE SOCIEDAD MERCANTIL."
)
A_P1_NAME = (
    'PRIMERA.- DENOMINACION: La sociedad se denominara "TECNOLOGIA FINANCIERA DEL VALLE", '
    "seguida de las palabras SOCIEDAD ANONIMA DE CAPITAL VARIABLE, o de su abreviatura "
    '"S.A. DE C.V."'
)
A_P1_CAPITAL = (
    "SEXTA.- CAPITAL SOCIAL: El capital social fijo es la cantidad de $500,000.00 "
    "(QUINIENTOS MIL PESOS 00/100 MONEDA NACIONAL), representado por 500 acciones "
    "ordinarias nominativas con valor nominal de $1,000.00 cada una."
)
A_P2_SHAREHOLDER_1 = (
    "1. CARLOS EDUARDO RAMIREZ SOTO, con RFC RASC850412HM1, suscribe 300 acciones con "
    "valor de $300,000.00 (60.0%)."
)
A_P2_SHAREHOLDER_2 = (
    "2. SOFIA VALENTINA HERRERA CRUZ, con RFC HECS901105MJ2, suscribe 200 acciones con "
    "valor de $200,000.00 (40.0%)."
)
A_P2_REP_1 = (
    "Se designa como ADMINISTRADOR UNICO al señor CARLOS EDUARDO RAMIREZ SOTO, a quien se le "
    "confiere PODER GENERAL PARA PLEITOS Y COBRANZAS, PODER GENERAL PARA ACTOS DE "
    "ADMINISTRACION Y PODER GENERAL PARA ACTOS DE DOMINIO, para ser ejercido de manera "
    "INDIVIDUAL."
)
A_P2_REP_2 = (
    "Se designa como APODERADA GENERAL a la señora SOFIA VALENTINA HERRERA CRUZ, a quien se le "
    "confiere PODER GENERAL PARA PLEITOS Y COBRANZAS Y ACTOS DE ADMINISTRACION, para ser "
    "ejercido de manera INDIVIDUAL."
)

B_P1_HEADER = "ESCRITURA PUBLICA NUMERO 12,890 (DOCE MIL OCHOCIENTOS NOVENTA)."
B_P1_DATE_PLACE = (
    "En la Ciudad de Monterrey, Estado de Nuevo Leon, a 22 de julio de 2020, ante mi, "
    "Licenciado FERNANDO GARZA TREVINO, Titular de la Notaria Publica Numero 42 del Primer "
    "Distrito Registral en el Estado de Nuevo Leon."
)
B_P1_NAME = (
    'CLAUSULA PRIMERA.- La sociedad mercantil adoptara la denominacion de '
    '"COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. DE C.V."'
)
B_P2_CAPITAL = (
    "QUINTA.- El capital social es de $250,000.00 (DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.)."
)
B_P2_SHAREHOLDERS = (
    "JORGE ALBERTO SALINAS MORA (150 partes sociales, $150,000.00, 60.0%) y PATRICIA EUGENIA "
    "LOZANO VELA (100 partes sociales, $100,000.00, 40.0%)."
)
B_P2_REPRESENTATIVE = (
    "GERENTE GENERAL: JORGE ALBERTO SALINAS MORA con PODER GENERAL PARA PLEITOS Y COBRANZAS Y "
    "ACTOS DE ADMINISTRACION (Individual)."
)

C_P1_HEADER = "ESCRITURA PUBLICA NUMERO 30,100 (TREINTA MIL CIEN)."
C_P1_DATE_PLACE = (
    "En la Ciudad de Guadalajara, Jalisco, a 10 de mayo de 2019, ante el Licenciado ARTURO "
    "VILLASEÑOR ORTIZ, Notario Publico Numero 15 de Guadalajara, Jalisco, se constituye "
    '"LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V."'
)
C_P1_CAPITAL = (
    "CLAUSULA QUINTA (CAPITAL ORIGINAL): El capital social inicial se fija en la suma de "
    "$100,000.00 (CIEN MIL PESOS 00/100 MONEDA NACIONAL)."
)
C_P1_SHAREHOLDERS = (
    "GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V. (800 acciones, $80,000.00, 80.0%) y "
    "MARIA ELENA TORRES VEGA (200 acciones, $20,000.00, 20.0%)."
)
C_P2_HEADING = "TRANSITORIA SEGUNDA (NOMBRAMIENTOS ORIGINALES 2019):"
C_P2_REP_1 = (
    "1. Se nombra como ADMINISTRADOR UNICO al LIC. JUAN CARLOS PÉREZ LÓPEZ, otorgandole PODER "
    "GENERAL PARA PLEITOS Y COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO de forma "
    "INDIVIDUAL."
)
C_P2_REP_2 = (
    "2. Se nombra como APODERADA LEGAL a MARIA ELENA TORRES VEGA, otorgandole PODER GENERAL "
    "PARA PLEITOS Y COBRANZAS Y ACTOS DE ADMINISTRACION de forma INDIVIDUAL."
)
C_P3_HEADER = "ESCRITURA PUBLICA NUMERO 58,400 (CINCUENTA Y OCHO MIL CUATROCIENTOS)."
C_P3_DATE_PLACE = (
    "En la Ciudad de Guadalajara, Jalisco, a 20 de agosto de 2024, ante el Licenciado ARTURO "
    "VILLASEÑOR ORTIZ, Notario Publico Numero 15 de Guadalajara, comparece el Delegado "
    "Especial para protocolizar el Acta de Asamblea General Extraordinaria de Accionistas de "
    '"LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V."'
)
C_P3_CAPITAL_INCREASE = (
    "PRIMER ACUERDO (AUMENTO DE CAPITAL SOCIAL): Se aprueba aumentar el capital social fijo "
    "de la suma de $100,000.00 M.N. para quedar fijado en la cantidad actual de "
    "$1,000,000.00 (UN MILLON DE PESOS 00/100 MONEDA NACIONAL)."
)
C_P3_REVOCATION = (
    "SEGUNDO ACUERDO (REVOCACION DE PODERES Y CARGO): Se revoca en este acto el nombramiento "
    "de Administrador Unico, asi como todos y cada uno de los poderes generales para pleitos y "
    "cobranzas, actos de administracion y actos de dominio otorgados previamente en la "
    "escritura 30,100 a favor del SR. JUAN C. PEREZ LOPEZ, quedando sin efecto legal alguno su "
    "representacion."
)
C_P4_NEW_APPOINTMENT = (
    "TERCER ACUERDO (NUEVO NOMBRAMIENTO Y OTORGAMIENTO DE PODERES): Se designa como nuevo "
    "ADMINISTRADOR UNICO de la sociedad al ING. ALEJANDRO GÓMEZ RUIZ, a quien se le confiere "
    "PODER GENERAL PARA PLEITOS Y COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO para "
    "ser ejercido de manera INDIVIDUAL. Se ratifican los poderes vigentes de MARIA ELENA "
    "TORRES VEGA."
)


def _body(text: str, mandated: bool = False) -> TextBlock:
    """Build a body paragraph.

    Parameters
    ----------
    text : str
        Paragraph text.
    mandated : bool, default False
        Whether spec 02 mandates the exact text.

    Returns
    -------
    TextBlock
        Body-styled block.
    """
    return TextBlock(text=text, style=BlockStyle.BODY, mandated=mandated)


def _heading(text: str, mandated: bool = False) -> TextBlock:
    """Build a section heading.

    Parameters
    ----------
    text : str
        Heading text.
    mandated : bool, default False
        Whether spec 02 mandates the exact text.

    Returns
    -------
    TextBlock
        Heading-styled block.
    """
    return TextBlock(text=text, style=BlockStyle.HEADING, mandated=mandated)


def _title(text: str, mandated: bool = False) -> TextBlock:
    """Build a deed title line.

    Parameters
    ----------
    text : str
        Title text.
    mandated : bool, default False
        Whether spec 02 mandates the exact text.

    Returns
    -------
    TextBlock
        Title-styled block.
    """
    return TextBlock(text=text, style=BlockStyle.TITLE, mandated=mandated)


def _signature(text: str) -> TextBlock:
    """Build a signature block line.

    Parameters
    ----------
    text : str
        Signature caption.

    Returns
    -------
    TextBlock
        Signature-styled block.
    """
    return TextBlock(text=text, style=BlockStyle.SIGNATURE)


# ---------------------------------------------------------------------------
# Page layouts
# ---------------------------------------------------------------------------

ACTA_A = DeedSpec(
    acta_id="acta_a",
    filename="acta_a.pdf",
    title="Instrumento Publico 45,120 - Tecnologia Financiera del Valle",
    scenario="Clean digital native incorporation deed (2 native pages).",
    pages=(
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 88 - CIUDAD DE MEXICO",
            footer="Hoja 1 de 2 - Testimonio del Instrumento Publico 45,120",
            blocks=(
                _title(A_P1_HEADER, mandated=True),
                _heading("LIBRO OCHOCIENTOS DIECISIETE - CONTRATO DE SOCIEDAD"),
                _body(A_P1_DATE_PLACE, mandated=True),
                _body(
                    "Comparecen los otorgantes, quienes manifiestan su voluntad de constituir una "
                    "sociedad anonima de capital variable conforme a la Ley General de Sociedades "
                    "Mercantiles, sujeta a los siguientes:"
                ),
                _heading("ESTATUTOS SOCIALES"),
                _body(A_P1_NAME, mandated=True),
                _body(
                    "SEGUNDA.- OBJETO: La sociedad tendra por objeto el desarrollo, licenciamiento "
                    "y operacion de plataformas tecnologicas para servicios financieros, asi como "
                    "la prestacion de servicios de consultoria en sistemas de informacion."
                ),
                _body(
                    "TERCERA.- DURACION: La duracion de la sociedad sera de noventa y nueve "
                    "anualidades contadas a partir de la firma del presente instrumento."
                ),
                _body(
                    "CUARTA.- DOMICILIO: El domicilio social sera la Ciudad de Mexico, sin perjuicio "
                    "de establecer oficinas o sucursales en cualquier otro lugar de la Republica."
                ),
                _body(
                    "QUINTA.- NACIONALIDAD: La sociedad es mexicana y se conviene la clausula de "
                    "admision de extranjeros en los terminos de la Ley de Inversion Extranjera."
                ),
                _body(A_P1_CAPITAL, mandated=True),
                _body(
                    "SEPTIMA.- ADMINISTRACION: La direccion y administracion de la sociedad estara "
                    "a cargo de un Administrador Unico o de un Consejo de Administracion, segun lo "
                    "determine la Asamblea General de Accionistas."
                ),
            ),
        ),
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 88 - CIUDAD DE MEXICO",
            footer="Hoja 2 de 2 - Testimonio del Instrumento Publico 45,120",
            blocks=(
                _heading("CLAUSULAS TRANSITORIAS"),
                _body(
                    "TRANSITORIA PRIMERA.- SUSCRIPCION Y PAGO: El capital social fijo queda "
                    "integramente suscrito y pagado por los accionistas de la siguiente manera:"
                ),
                _body(A_P2_SHAREHOLDER_1, mandated=True),
                _body(A_P2_SHAREHOLDER_2, mandated=True),
                _body(
                    "TRANSITORIA SEGUNDA.- NOMBRAMIENTOS Y PODERES: Los accionistas acuerdan por "
                    "unanimidad los siguientes nombramientos:"
                ),
                _body(A_P2_REP_1, mandated=True),
                _body(A_P2_REP_2, mandated=True),
                _body(
                    "YO, EL NOTARIO, DOY FE de la identidad de los comparecientes, de que les lei "
                    "integramente el presente instrumento y de que, conformes con su contenido, lo "
                    "ratifican y firman ante mi."
                ),
                _signature("LIC. ROBERTO MENDEZ LARA"),
                _signature("TITULAR DE LA NOTARIA PUBLICA NUMERO 88 DE LA CIUDAD DE MEXICO"),
            ),
        ),
    ),
)

ACTA_B = DeedSpec(
    acta_id="acta_b",
    filename="acta_b.pdf",
    title="Escritura Publica 12,890 - Comercializadora Industrial del Norte",
    scenario=(
        "Hybrid deed: native page 1 and scanned page 2 with a notarial seal over the "
        "capital clause (OCR 0.72, verifier p=0.78)."
    ),
    pages=(
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 42 - MONTERREY, NUEVO LEON",
            footer="Hoja 1 de 2 - Escritura Publica 12,890 - Continua al reverso",
            blocks=(
                _title(B_P1_HEADER, mandated=True),
                _heading("CONSTITUCION DE SOCIEDAD DE RESPONSABILIDAD LIMITADA"),
                _body(B_P1_DATE_PLACE, mandated=True),
                _body(
                    "Comparecen los socios fundadores con el objeto de constituir una sociedad de "
                    "responsabilidad limitada de capital variable, la cual se regira por los "
                    "siguientes:"
                ),
                _heading("ESTATUTOS SOCIALES"),
                _body(B_P1_NAME, mandated=True),
                _body(
                    "CLAUSULA SEGUNDA.- OBJETO: La sociedad tendra por objeto la fabricacion, "
                    "transformacion y"
                ),
            ),
        ),
        RasterPageSpec(
            regions=(
                RasterRegionSpec(
                    text="HOJA DOS (2) - CONTINUACION DE LOS ESTATUTOS SOCIALES",
                    bbox=(72.0, 72.0, 540.0, 100.0),
                    ocr_confidence=0.95,
                ),
                RasterRegionSpec(
                    text=(
                        "comercializacion de toda clase de productos industriales, maquinaria y "
                        "refacciones, asi como la celebracion de todos los actos y contratos "
                        "permitidos por la ley."
                    ),
                    bbox=(72.0, 120.0, 540.0, 175.0),
                    ocr_confidence=0.93,
                ),
                RasterRegionSpec(
                    text=(
                        "CLAUSULA TERCERA.- DURACION: La duracion de la sociedad sera indefinida."
                    ),
                    bbox=(72.0, 195.0, 540.0, 230.0),
                    ocr_confidence=0.94,
                ),
                RasterRegionSpec(
                    text=(
                        "CLAUSULA CUARTA.- DOMICILIO: El domicilio social sera la Ciudad de "
                        "Monterrey, Nuevo Leon, sin perjuicio de establecer sucursales en otros "
                        "lugares."
                    ),
                    bbox=(72.0, 250.0, 540.0, 305.0),
                    ocr_confidence=0.94,
                ),
                RasterRegionSpec(
                    text=(
                        "Los socios fundadores convienen que el capital social quedara integrado "
                        "conforme a la clausula siguiente."
                    ),
                    bbox=(72.0, 330.0, 540.0, 385.0),
                    ocr_confidence=0.93,
                ),
                RasterRegionSpec(
                    text=B_P2_CAPITAL,
                    bbox=ACTA_B_CAPITAL_BBOX,
                    ocr_confidence=ACTA_B_CAPITAL_OCR_CONFIDENCE,
                    mandated=True,
                ),
                RasterRegionSpec(
                    text="CLAUSULA SEXTA.- PARTES SOCIALES: " + B_P2_SHAREHOLDERS,
                    bbox=(72.0, 480.0, 540.0, 545.0),
                    ocr_confidence=0.94,
                    mandated=True,
                ),
                RasterRegionSpec(
                    text="CLAUSULA SEPTIMA.- ADMINISTRACION: " + B_P2_REPRESENTATIVE,
                    bbox=(72.0, 565.0, 540.0, 615.0),
                    ocr_confidence=0.95,
                    mandated=True,
                ),
            ),
            seal_text=ACTA_B_SEAL_TEXT,
            seal_center_bbox=ACTA_B_CAPITAL_BBOX,
            noise_seed=ACTA_B_NOISE_SEED,
        ),
    ),
)

ACTA_C = DeedSpec(
    acta_id="acta_c",
    filename="acta_c.pdf",
    title="Escrituras 30,100 y 58,400 - Logistica Transcontinental del Pacifico",
    scenario=(
        "Multi-act file: 2019 incorporation (pages 1-2) and 2024 extraordinary assembly "
        "protocolization (pages 3-4) with capital increase, revocation and new appointment."
    ),
    pages=(
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 15 - GUADALAJARA, JALISCO",
            footer="Hoja 1 de 4 - Escritura Publica 30,100 (Constitucion)",
            blocks=(
                _title(C_P1_HEADER, mandated=True),
                _heading("CONSTITUCION DE SOCIEDAD ANONIMA DE CAPITAL VARIABLE"),
                _body(C_P1_DATE_PLACE, mandated=True),
                _heading("ESTATUTOS SOCIALES"),
                _body(
                    "CLAUSULA SEGUNDA (OBJETO): La sociedad tendra por objeto la prestacion de "
                    "servicios de transporte multimodal, almacenaje, consolidacion de carga y "
                    "logistica internacional."
                ),
                _body(
                    "CLAUSULA TERCERA (DOMICILIO): El domicilio social sera la Ciudad de "
                    "Guadalajara, Jalisco."
                ),
                _body(C_P1_CAPITAL, mandated=True),
                _body(
                    "CLAUSULA SEXTA (SUSCRIPCION): El capital social inicial queda suscrito y "
                    "pagado por los accionistas fundadores de la siguiente manera:"
                ),
                _body(C_P1_SHAREHOLDERS, mandated=True),
            ),
        ),
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 15 - GUADALAJARA, JALISCO",
            footer="Hoja 2 de 4 - Escritura Publica 30,100 (Constitucion)",
            blocks=(
                _heading("CLAUSULAS TRANSITORIAS"),
                _body(
                    "TRANSITORIA PRIMERA: La sociedad iniciara sus operaciones a partir de la fecha "
                    "de firma de la presente escritura."
                ),
                _heading(C_P2_HEADING, mandated=True),
                _body(C_P2_REP_1, mandated=True),
                _body(C_P2_REP_2, mandated=True),
                _body(
                    "YO, EL NOTARIO, DOY FE de que conozco a los comparecientes, de que les lei la "
                    "presente escritura y de que la firman en mi presencia."
                ),
                _signature("LIC. ARTURO VILLASEÑOR ORTIZ"),
                _signature("NOTARIO PUBLICO NUMERO 15 DE GUADALAJARA, JALISCO"),
            ),
        ),
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 15 - GUADALAJARA, JALISCO",
            footer="Hoja 3 de 4 - Escritura Publica 58,400 (Protocolizacion de Asamblea)",
            blocks=(
                _title(C_P3_HEADER, mandated=True),
                _heading("PROTOCOLIZACION DE ACTA DE ASAMBLEA GENERAL EXTRAORDINARIA"),
                _body(C_P3_DATE_PLACE, mandated=True),
                _body(
                    "El Delegado Especial exhibe el acta de la asamblea celebrada con la asistencia "
                    "de la totalidad de las acciones representativas del capital social, en la que "
                    "se adoptaron por unanimidad los siguientes:"
                ),
                _heading("ACUERDOS"),
                _body(C_P3_CAPITAL_INCREASE, mandated=True),
                _body(C_P3_REVOCATION, mandated=True),
            ),
        ),
        NativePageSpec(
            running_header="NOTARIA PUBLICA NUMERO 15 - GUADALAJARA, JALISCO",
            footer="Hoja 4 de 4 - Escritura Publica 58,400 (Protocolizacion de Asamblea)",
            blocks=(
                _body(C_P4_NEW_APPOINTMENT, mandated=True),
                _body(
                    "CUARTO ACUERDO (DELEGADO ESPECIAL): Se designa Delegado Especial de la asamblea "
                    "para que acuda ante el Notario Publico de su eleccion a protocolizar la "
                    "presente acta y gestionar su inscripcion en el Registro Publico de Comercio."
                ),
                _body(
                    "YO, EL NOTARIO, DOY FE de que tuve a la vista el acta original de la asamblea, "
                    "de que el compareciente acredito su caracter de Delegado Especial y de que "
                    "firma la presente escritura en mi presencia."
                ),
                _signature("LIC. ARTURO VILLASEÑOR ORTIZ"),
                _signature("NOTARIO PUBLICO NUMERO 15 DE GUADALAJARA, JALISCO"),
            ),
        ),
    ),
)

DEED_SPECS: tuple[DeedSpec, ...] = (ACTA_A, ACTA_B, ACTA_C)
"""All synthetic deeds in generation order."""


# ---------------------------------------------------------------------------
# Raw extraction builders (pre-resolution, Appendix B semantics)
# ---------------------------------------------------------------------------


def _bbox(coordinates: Optional[tuple[float, float, float, float]]) -> Optional[BoundingBox]:
    """Convert an optional coordinate tuple into a bounding box.

    Parameters
    ----------
    coordinates : tuple of float or None
        ``(x1, y1, x2, y2)`` in PDF points.

    Returns
    -------
    BoundingBox or None
        The box, or ``None`` when no coordinates are given.
    """
    if coordinates is None:
        return None
    return BoundingBox.from_sequence(list(coordinates))


def _native(page: int, quote: str) -> Evidence:
    """Build native-PDF evidence (confidence 1.0).

    Parameters
    ----------
    page : int
        1-indexed page number.
    quote : str
        Verbatim substring of the page text.

    Returns
    -------
    Evidence
        Native evidence.
    """
    return Evidence(
        page_number=page,
        source_quote=quote,
        confidence_score=1.0,
        confidence_source=ConfidenceSource.NATIVE_PDF,
    )


def _ocr(page: int, quote: str, region: RasterRegionSpec) -> Evidence:
    """Build OCR-engine evidence carrying the region confidence and bbox.

    Parameters
    ----------
    page : int
        1-indexed page number.
    quote : str
        Verbatim substring of the OCR text.
    region : RasterRegionSpec
        Region the quote was read from.

    Returns
    -------
    Evidence
        OCR evidence with bounding box.
    """
    return Evidence(
        page_number=page,
        source_quote=quote,
        confidence_score=region.ocr_confidence,
        confidence_source=ConfidenceSource.OCR_ENGINE,
        bounding_box=_bbox(region.bbox),
    )


def _extracted(value: Any, evidence: Evidence) -> dict[str, Any]:
    """Build the keyword arguments of an extracted field.

    Parameters
    ----------
    value : Any
        Field value.
    evidence : Evidence
        Primary evidence.

    Returns
    -------
    dict of str to Any
        Mapping accepted by :class:`FieldWithEvidence`.
    """
    return {"value": value, "status": FieldStatus.EXTRACTED, "evidence": evidence}


def _powers(
    grantee: str, power_types: tuple[str, ...], evidence: Evidence
) -> list[PowerOfAttorneyItem]:
    """Expand a grant into one item per power type.

    Parameters
    ----------
    grantee : str
        Grantee name as written in the deed.
    power_types : tuple of str
        Canonical power type strings.
    evidence : Evidence
        Evidence shared by all items of the grant.

    Returns
    -------
    list of PowerOfAttorneyItem
        One item per power type.
    """
    return [
        PowerOfAttorneyItem(
            otorgado_a=grantee,
            tipo_poder=power_type,
            modalidad=MODALITY_INDIVIDUAL,
            vigente=True,
            evidence=evidence.model_copy(deep=True),
        )
        for power_type in power_types
    ]


ALL_POWERS: tuple[str, ...] = (POWER_LITIGATION, POWER_ADMINISTRATION, POWER_OWNERSHIP)
"""The three general powers in canonical order."""

LITIGATION_AND_ADMINISTRATION: tuple[str, ...] = (POWER_LITIGATION, POWER_ADMINISTRATION)
"""Litigation plus administration powers."""


def build_acta_a_raw() -> RawDeedExtraction:
    """Build the pre-verified raw extraction for ``acta_a.pdf``.

    Returns
    -------
    RawDeedExtraction
        Validated raw extraction.
    """
    socio_1 = _native(2, A_P2_SHAREHOLDER_1.rstrip("."))
    socio_2 = _native(2, A_P2_SHAREHOLDER_2.rstrip("."))
    rep_1 = _native(2, "Se designa como ADMINISTRADOR UNICO al señor CARLOS EDUARDO RAMIREZ SOTO")
    rep_2 = _native(2, "Se designa como APODERADA GENERAL a la señora SOFIA VALENTINA HERRERA CRUZ")
    grant_1 = _native(2, A_P2_REP_1.rstrip("."))
    grant_2 = _native(2, A_P2_REP_2.rstrip("."))
    return RawDeedExtraction(
        razon_social=FieldWithEvidence[str](
            **_extracted(
                "TECNOLOGIA FINANCIERA DEL VALLE, S.A. DE C.V.",
                _native(
                    1,
                    'La sociedad se denominara "TECNOLOGIA FINANCIERA DEL VALLE", seguida de las '
                    "palabras SOCIEDAD ANONIMA DE CAPITAL VARIABLE",
                ),
            )
        ),
        fecha_constitucion=FieldWithEvidence[str](
            **_extracted(
                "2021-03-15",
                _native(1, "En la Ciudad de Mexico, a los 15 dias del mes de marzo de 2021"),
            )
        ),
        numero_escritura=FieldWithEvidence[str](
            **_extracted(
                "45,120",
                _native(1, "INSTRUMENTO PUBLICO NUMERO 45,120 (CUARENTA Y CINCO MIL CIENTO VEINTE)"),
            )
        ),
        notario=FieldWithEvidence[NotaryInfo](
            **_extracted(
                NotaryInfo(
                    nombre="ROBERTO MENDEZ LARA",
                    numero_notaria=88,
                    entidad_federativa="Ciudad de Mexico",
                ),
                _native(
                    1,
                    "el Licenciado ROBERTO MENDEZ LARA, Titular de la Notaria Publica Numero 88 de "
                    "la Ciudad de Mexico",
                ),
            )
        ),
        socios=FieldWithEvidence[list[ShareholderItem]](
            **_extracted(
                [
                    ShareholderItem(
                        nombre="CARLOS EDUARDO RAMIREZ SOTO",
                        rfc="RASC850412HM1",
                        acciones=300,
                        valor_aportacion=300000.0,
                        porcentaje=60.0,
                        evidence=socio_1,
                    ),
                    ShareholderItem(
                        nombre="SOFIA VALENTINA HERRERA CRUZ",
                        rfc="HECS901105MJ2",
                        acciones=200,
                        valor_aportacion=200000.0,
                        porcentaje=40.0,
                        evidence=socio_2,
                    ),
                ],
                socio_1.model_copy(deep=True),
            )
        ),
        representantes_legales=FieldWithEvidence[list[LegalRepresentativeItem]](
            **_extracted(
                [
                    LegalRepresentativeItem(
                        nombre="CARLOS EDUARDO RAMIREZ SOTO",
                        nombre_normalizado="CARLOS EDUARDO RAMIREZ SOTO",
                        cargo="Administrador Unico",
                        vigente=True,
                        fecha_nombramiento="2021-03-15",
                        evidence=rep_1,
                    ),
                    LegalRepresentativeItem(
                        nombre="SOFIA VALENTINA HERRERA CRUZ",
                        nombre_normalizado="SOFIA VALENTINA HERRERA CRUZ",
                        cargo="Apoderada General",
                        vigente=True,
                        fecha_nombramiento="2021-03-15",
                        evidence=rep_2,
                    ),
                ],
                rep_1.model_copy(deep=True),
            )
        ),
        poderes=FieldWithEvidence[list[PowerOfAttorneyItem]](
            **_extracted(
                _powers("CARLOS EDUARDO RAMIREZ SOTO", ALL_POWERS, grant_1)
                + _powers("SOFIA VALENTINA HERRERA CRUZ", LITIGATION_AND_ADMINISTRATION, grant_2),
                grant_1.model_copy(deep=True),
            )
        ),
        capital_social_actual=FieldWithEvidence[CapitalInfo](
            **_extracted(
                CapitalInfo(
                    monto_numerico=500000.0,
                    moneda="MXN",
                    monto_en_letra="QUINIENTOS MIL PESOS 00/100 MONEDA NACIONAL",
                    fecha_acto_vigente="2021-03-15",
                    escritura_origen="45,120",
                ),
                _native(
                    1,
                    "El capital social fijo es la cantidad de $500,000.00 (QUINIENTOS MIL PESOS "
                    "00/100 MONEDA NACIONAL)",
                ),
            )
        ),
        amendment_history=[
            AmendmentEvent(
                event_type=AmendmentType.INCORPORATION,
                fecha_acto="2021-03-15",
                numero_escritura="45,120",
                descripcion=(
                    "Original incorporation of TECNOLOGIA FINANCIERA DEL VALLE, S.A. DE C.V. "
                    "before Notary Public 88 of Ciudad de Mexico."
                ),
                monto_nuevo=500000.0,
                evidence=_native(1, "hago constar el CONTRATO DE SOCIEDAD MERCANTIL"),
            )
        ],
    )


def acta_b_region(fragment: str) -> RasterRegionSpec:
    """Return the Acta B scanned-page region containing a text fragment.

    Parameters
    ----------
    fragment : str
        Verbatim text expected inside exactly one region.

    Returns
    -------
    RasterRegionSpec
        The unique region whose text contains ``fragment``.

    Raises
    ------
    TypeError
        If Acta B page 2 is not a raster page.
    LookupError
        If zero or several regions contain the fragment.
    """
    page = ACTA_B.pages[1]
    if not isinstance(page, RasterPageSpec):
        raise TypeError("acta_b page 2 must be a raster page")
    matches = [region for region in page.regions if fragment in region.text]
    if len(matches) != 1:
        raise LookupError(f"expected one acta_b region containing {fragment!r}, found {len(matches)}")
    return matches[0]


def build_acta_b_raw() -> RawDeedExtraction:
    """Build the pre-verified raw extraction for ``acta_b.pdf``.

    Returns
    -------
    RawDeedExtraction
        Validated raw extraction; page-2 evidence uses OCR confidences.
    """
    capital_region = acta_b_region(B_P2_CAPITAL)
    socios_region = acta_b_region(B_P2_SHAREHOLDERS)
    rep_region = acta_b_region(B_P2_REPRESENTATIVE)
    socio_1 = _ocr(
        2, "JORGE ALBERTO SALINAS MORA (150 partes sociales, $150,000.00, 60.0%)", socios_region
    )
    socio_2 = _ocr(
        2, "PATRICIA EUGENIA LOZANO VELA (100 partes sociales, $100,000.00, 40.0%)", socios_region
    )
    rep = _ocr(2, "GERENTE GENERAL: JORGE ALBERTO SALINAS MORA", rep_region)
    grant = _ocr(2, B_P2_REPRESENTATIVE.rstrip("."), rep_region)
    return RawDeedExtraction(
        razon_social=FieldWithEvidence[str](
            **_extracted(
                "COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. DE C.V.",
                _native(
                    1,
                    'adoptara la denominacion de "COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE '
                    'R.L. DE C.V."',
                ),
            )
        ),
        fecha_constitucion=FieldWithEvidence[str](
            **_extracted(
                "2020-07-22",
                _native(1, "En la Ciudad de Monterrey, Estado de Nuevo Leon, a 22 de julio de 2020"),
            )
        ),
        numero_escritura=FieldWithEvidence[str](
            **_extracted(
                "12,890",
                _native(1, "ESCRITURA PUBLICA NUMERO 12,890 (DOCE MIL OCHOCIENTOS NOVENTA)"),
            )
        ),
        notario=FieldWithEvidence[NotaryInfo](
            **_extracted(
                NotaryInfo(
                    nombre="FERNANDO GARZA TREVINO",
                    numero_notaria=42,
                    entidad_federativa="Nuevo Leon",
                ),
                _native(
                    1,
                    "Licenciado FERNANDO GARZA TREVINO, Titular de la Notaria Publica Numero 42 del "
                    "Primer Distrito Registral en el Estado de Nuevo Leon",
                ),
            )
        ),
        socios=FieldWithEvidence[list[ShareholderItem]](
            **_extracted(
                [
                    ShareholderItem(
                        nombre="JORGE ALBERTO SALINAS MORA",
                        rfc=None,
                        acciones=150,
                        valor_aportacion=150000.0,
                        porcentaje=60.0,
                        evidence=socio_1,
                    ),
                    ShareholderItem(
                        nombre="PATRICIA EUGENIA LOZANO VELA",
                        rfc=None,
                        acciones=100,
                        valor_aportacion=100000.0,
                        porcentaje=40.0,
                        evidence=socio_2,
                    ),
                ],
                socio_1.model_copy(deep=True),
            )
        ),
        representantes_legales=FieldWithEvidence[list[LegalRepresentativeItem]](
            **_extracted(
                [
                    LegalRepresentativeItem(
                        nombre="JORGE ALBERTO SALINAS MORA",
                        nombre_normalizado="JORGE ALBERTO SALINAS MORA",
                        cargo="Gerente General",
                        vigente=True,
                        fecha_nombramiento="2020-07-22",
                        evidence=rep,
                    )
                ],
                rep.model_copy(deep=True),
            )
        ),
        poderes=FieldWithEvidence[list[PowerOfAttorneyItem]](
            **_extracted(
                _powers("JORGE ALBERTO SALINAS MORA", LITIGATION_AND_ADMINISTRATION, grant),
                grant.model_copy(deep=True),
            )
        ),
        capital_social_actual=FieldWithEvidence[CapitalInfo](
            **_extracted(
                CapitalInfo(
                    monto_numerico=250000.0,
                    moneda="MXN",
                    monto_en_letra="DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.",
                    fecha_acto_vigente="2020-07-22",
                    escritura_origen="12,890",
                ),
                _ocr(2, B_P2_CAPITAL, capital_region),
            )
        ),
        amendment_history=[
            AmendmentEvent(
                event_type=AmendmentType.INCORPORATION,
                fecha_acto="2020-07-22",
                numero_escritura="12,890",
                descripcion=(
                    "Original incorporation of COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. "
                    "DE C.V. before Notary Public 42 of Nuevo Leon."
                ),
                monto_nuevo=250000.0,
                evidence=_native(1, "ESCRITURA PUBLICA NUMERO 12,890 (DOCE MIL OCHOCIENTOS NOVENTA)"),
            )
        ],
    )


def build_acta_c_raw() -> RawDeedExtraction:
    """Build the pre-verified raw extraction for ``acta_c.pdf``.

    Returns
    -------
    RawDeedExtraction
        Validated raw extraction. Representatives and powers include the
        later-revoked Juan Carlos Perez Lopez; capital is the original
        incorporation capital (the resolver selects the 2024 increase).
    """
    socio_1 = _native(1, "GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V. (800 acciones, $80,000.00, 80.0%)")
    socio_2 = _native(1, "MARIA ELENA TORRES VEGA (200 acciones, $20,000.00, 20.0%)")
    rep_1 = _native(2, "Se nombra como ADMINISTRADOR UNICO al LIC. JUAN CARLOS PÉREZ LÓPEZ")
    rep_2 = _native(2, "Se nombra como APODERADA LEGAL a MARIA ELENA TORRES VEGA")
    rep_3 = _native(4, "Se designa como nuevo ADMINISTRADOR UNICO de la sociedad al ING. ALEJANDRO GÓMEZ RUIZ")
    grant_1 = _native(
        2,
        "ADMINISTRADOR UNICO al LIC. JUAN CARLOS PÉREZ LÓPEZ, otorgandole PODER GENERAL PARA "
        "PLEITOS Y COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO de forma INDIVIDUAL",
    )
    grant_2 = _native(
        2,
        "APODERADA LEGAL a MARIA ELENA TORRES VEGA, otorgandole PODER GENERAL PARA PLEITOS Y "
        "COBRANZAS Y ACTOS DE ADMINISTRACION de forma INDIVIDUAL",
    )
    grant_3 = _native(
        4,
        "al ING. ALEJANDRO GÓMEZ RUIZ, a quien se le confiere PODER GENERAL PARA PLEITOS Y "
        "COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO para ser ejercido de manera "
        "INDIVIDUAL",
    )
    return RawDeedExtraction(
        razon_social=FieldWithEvidence[str](
            **_extracted(
                "LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V.",
                _native(1, 'se constituye "LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V."'),
            )
        ),
        fecha_constitucion=FieldWithEvidence[str](
            **_extracted(
                "2019-05-10",
                _native(1, "En la Ciudad de Guadalajara, Jalisco, a 10 de mayo de 2019"),
            )
        ),
        numero_escritura=FieldWithEvidence[str](
            **_extracted("30,100", _native(1, "ESCRITURA PUBLICA NUMERO 30,100 (TREINTA MIL CIEN)"))
        ),
        notario=FieldWithEvidence[NotaryInfo](
            **_extracted(
                NotaryInfo(
                    nombre="ARTURO VILLASEÑOR ORTIZ",
                    numero_notaria=15,
                    entidad_federativa="Jalisco",
                ),
                _native(
                    1,
                    "ante el Licenciado ARTURO VILLASEÑOR ORTIZ, Notario Publico Numero 15 de "
                    "Guadalajara, Jalisco",
                ),
            )
        ),
        socios=FieldWithEvidence[list[ShareholderItem]](
            **_extracted(
                [
                    ShareholderItem(
                        nombre="GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V.",
                        rfc=None,
                        acciones=800,
                        valor_aportacion=80000.0,
                        porcentaje=80.0,
                        evidence=socio_1,
                    ),
                    ShareholderItem(
                        nombre="MARIA ELENA TORRES VEGA",
                        rfc=None,
                        acciones=200,
                        valor_aportacion=20000.0,
                        porcentaje=20.0,
                        evidence=socio_2,
                    ),
                ],
                socio_1.model_copy(deep=True),
            )
        ),
        representantes_legales=FieldWithEvidence[list[LegalRepresentativeItem]](
            **_extracted(
                [
                    LegalRepresentativeItem(
                        nombre="LIC. JUAN CARLOS PÉREZ LÓPEZ",
                        nombre_normalizado="JUAN CARLOS PEREZ LOPEZ",
                        cargo="Administrador Unico",
                        vigente=True,
                        fecha_nombramiento="2019-05-10",
                        evidence=rep_1,
                    ),
                    LegalRepresentativeItem(
                        nombre="MARIA ELENA TORRES VEGA",
                        nombre_normalizado="MARIA ELENA TORRES VEGA",
                        cargo="Apoderada Legal",
                        vigente=True,
                        fecha_nombramiento="2019-05-10",
                        evidence=rep_2,
                    ),
                    LegalRepresentativeItem(
                        nombre="ING. ALEJANDRO GÓMEZ RUIZ",
                        nombre_normalizado="ALEJANDRO GOMEZ RUIZ",
                        cargo="Administrador Unico",
                        vigente=True,
                        fecha_nombramiento="2024-08-20",
                        evidence=rep_3,
                    ),
                ],
                rep_1.model_copy(deep=True),
            )
        ),
        poderes=FieldWithEvidence[list[PowerOfAttorneyItem]](
            **_extracted(
                _powers("LIC. JUAN CARLOS PÉREZ LÓPEZ", ALL_POWERS, grant_1)
                + _powers("MARIA ELENA TORRES VEGA", LITIGATION_AND_ADMINISTRATION, grant_2)
                + _powers("ING. ALEJANDRO GÓMEZ RUIZ", ALL_POWERS, grant_3),
                grant_1.model_copy(deep=True),
            )
        ),
        capital_social_actual=FieldWithEvidence[CapitalInfo](
            **_extracted(
                CapitalInfo(
                    monto_numerico=100000.0,
                    moneda="MXN",
                    monto_en_letra="CIEN MIL PESOS 00/100 MONEDA NACIONAL",
                    fecha_acto_vigente="2019-05-10",
                    escritura_origen="30,100",
                ),
                _native(
                    1,
                    "El capital social inicial se fija en la suma de $100,000.00 (CIEN MIL PESOS "
                    "00/100 MONEDA NACIONAL)",
                ),
            )
        ),
        amendment_history=[
            AmendmentEvent(
                event_type=AmendmentType.INCORPORATION,
                fecha_acto="2019-05-10",
                numero_escritura="30,100",
                descripcion=(
                    "Original incorporation of LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE "
                    "C.V. with an initial capital of MXN 100,000.00."
                ),
                monto_nuevo=100000.0,
                evidence=_native(1, "ESCRITURA PUBLICA NUMERO 30,100 (TREINTA MIL CIEN)"),
            ),
            AmendmentEvent(
                event_type=AmendmentType.CAPITAL_INCREASE,
                fecha_acto="2024-08-20",
                numero_escritura="58,400",
                descripcion=(
                    "Extraordinary general assembly increases the fixed capital from MXN "
                    "100,000.00 to MXN 1,000,000.00."
                ),
                monto_nuevo=1000000.0,
                evidence=_native(
                    3,
                    "Se aprueba aumentar el capital social fijo de la suma de $100,000.00 M.N. para "
                    "quedar fijado en la cantidad actual de $1,000,000.00 (UN MILLON DE PESOS 00/100 "
                    "MONEDA NACIONAL)",
                ),
            ),
            AmendmentEvent(
                event_type=AmendmentType.REVOCATION,
                fecha_acto="2024-08-20",
                numero_escritura="58,400",
                descripcion=(
                    "Revocation of the sole administrator appointment and of all general powers "
                    "granted in deed 30,100."
                ),
                sujeto_afectado="SR. JUAN C. PEREZ LOPEZ",
                evidence=_native(
                    3,
                    "Se revoca en este acto el nombramiento de Administrador Unico, asi como todos y "
                    "cada uno de los poderes generales para pleitos y cobranzas, actos de "
                    "administracion y actos de dominio otorgados previamente en la escritura 30,100 "
                    "a favor del SR. JUAN C. PEREZ LOPEZ",
                ),
            ),
            AmendmentEvent(
                event_type=AmendmentType.GRANT_OF_POWERS,
                fecha_acto="2024-08-20",
                numero_escritura="58,400",
                descripcion=(
                    "Appointment of a new sole administrator with general powers for litigation, "
                    "administration and ownership."
                ),
                sujeto_afectado="ING. ALEJANDRO GÓMEZ RUIZ",
                evidence=_native(
                    4,
                    "Se designa como nuevo ADMINISTRADOR UNICO de la sociedad al ING. ALEJANDRO "
                    "GÓMEZ RUIZ, a quien se le confiere PODER GENERAL PARA PLEITOS Y COBRANZAS, "
                    "ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO",
                ),
            ),
        ],
    )


RAW_EXTRACTION_BUILDERS: dict[str, Callable[[], RawDeedExtraction]] = {
    "acta_a": build_acta_a_raw,
    "acta_b": build_acta_b_raw,
    "acta_c": build_acta_c_raw,
}
"""Raw extraction builder per acta identifier."""


# ---------------------------------------------------------------------------
# OCR and verifier cache entries
# ---------------------------------------------------------------------------


def build_ocr_result(page_number: int, page: RasterPageSpec) -> OCRResult:
    """Build the cached OCR result of a raster page.

    Parameters
    ----------
    page_number : int
        1-indexed page number.
    page : RasterPageSpec
        Raster page specification.

    Returns
    -------
    OCRResult
        Text (regions joined by newlines), regions and mean confidence.
    """
    regions = [
        OCRRegion(text=region.text, bbox=BoundingBox.from_sequence(list(region.bbox)), confidence=region.ocr_confidence)
        for region in page.regions
    ]
    mean_confidence = round(sum(region.confidence for region in regions) / len(regions), 4)
    return OCRResult(
        page_number=page_number,
        text="\n".join(region.text for region in page.regions),
        mean_confidence=mean_confidence,
        regions=regions,
        engine=OCR_ENGINE_NAME,
    )


def build_verifier_responses(spec: DeedSpec) -> list[dict[str, Any]]:
    """Build the cached Decisions predicate verifier responses.

    Parameters
    ----------
    spec : DeedSpec
        Deed specification.

    Returns
    -------
    list of dict
        Only Acta B has an entry (page 2 capital clause, ``p = 0.78``).
    """
    if spec.acta_id != "acta_b":
        return []
    return [
        {
            "page_number": 2,
            "bbox": list(ACTA_B_CAPITAL_BBOX),
            "predicate": ACTA_B_VERIFIER_PREDICATE,
            "probability": ACTA_B_VERIFIER_PROBABILITY,
        }
    ]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def _scalar(value: Any, page: int) -> dict[str, Any]:
    """Build a scalar ground-truth entry.

    Parameters
    ----------
    value : Any
        Expected value.
    page : int
        Expected evidence page.

    Returns
    -------
    dict of str to Any
        ``{"value": ..., "page": ...}``.
    """
    return {"value": value, "page": page}


def _power_entries(
    grantee_normalized: str, power_types: tuple[str, ...], page: int
) -> list[dict[str, Any]]:
    """Build expected active power entries for a grantee.

    Parameters
    ----------
    grantee_normalized : str
        Normalized grantee name.
    power_types : tuple of str
        Canonical power type strings.
    page : int
        Expected evidence page.

    Returns
    -------
    list of dict
        One entry per power type.
    """
    return [
        {
            "otorgado_a": grantee_normalized,
            "tipo_poder": power_type,
            "modalidad": MODALITY_INDIVIDUAL,
            "page": page,
        }
        for power_type in power_types
    ]


def _events(*events: tuple[str, str, str, int, Optional[str], Optional[float]]) -> dict[str, Any]:
    """Build the expected amendment history summary.

    Parameters
    ----------
    *events : tuple
        ``(event_type, fecha_acto, numero_escritura, page, sujeto_afectado,
        monto_nuevo)`` tuples in chronological order.

    Returns
    -------
    dict of str to Any
        ``event_count``, ``event_types`` and the detailed ``events`` list.
    """
    detailed = [
        {
            "event_type": event_type,
            "fecha_acto": fecha_acto,
            "numero_escritura": numero_escritura,
            "page": page,
            "sujeto_afectado": sujeto_afectado,
            "monto_nuevo": monto_nuevo,
        }
        for event_type, fecha_acto, numero_escritura, page, sujeto_afectado, monto_nuevo in events
    ]
    return {
        "event_count": len(detailed),
        "event_types": [entry["event_type"] for entry in detailed],
        "events": detailed,
    }


def build_ground_truth_fields(acta_id: str) -> dict[str, Any]:
    """Return the expected post-resolution values for one deed.

    Parameters
    ----------
    acta_id : str
        Deed identifier.

    Returns
    -------
    dict of str to Any
        ``fields``, ``revoked_representatives``,
        ``expected_human_review_fields`` and ``amendment_history``.

    Raises
    ------
    KeyError
        If ``acta_id`` is unknown.
    """
    incorporation = AmendmentType.INCORPORATION.value
    if acta_id == "acta_a":
        return {
            "fields": {
                "razon_social": _scalar("TECNOLOGIA FINANCIERA DEL VALLE, S.A. DE C.V.", 1),
                "fecha_constitucion": _scalar("2021-03-15", 1),
                "numero_escritura": _scalar("45,120", 1),
                "notario": _scalar(
                    {"nombre": "ROBERTO MENDEZ LARA", "numero_notaria": 88, "entidad_federativa": "Ciudad de Mexico"},
                    1,
                ),
                "socios": _scalar(
                    [
                        {"nombre": "CARLOS EDUARDO RAMIREZ SOTO", "rfc": "RASC850412HM1", "acciones": 300, "porcentaje": 60.0, "page": 2},
                        {"nombre": "SOFIA VALENTINA HERRERA CRUZ", "rfc": "HECS901105MJ2", "acciones": 200, "porcentaje": 40.0, "page": 2},
                    ],
                    2,
                ),
                "representantes_legales": _scalar(
                    [
                        {"nombre": "CARLOS EDUARDO RAMIREZ SOTO", "nombre_normalizado": "CARLOS EDUARDO RAMIREZ SOTO", "cargo": "Administrador Unico", "fecha_nombramiento": "2021-03-15", "page": 2},
                        {"nombre": "SOFIA VALENTINA HERRERA CRUZ", "nombre_normalizado": "SOFIA VALENTINA HERRERA CRUZ", "cargo": "Apoderada General", "fecha_nombramiento": "2021-03-15", "page": 2},
                    ],
                    2,
                ),
                "poderes": _scalar(
                    _power_entries("CARLOS EDUARDO RAMIREZ SOTO", ALL_POWERS, 2)
                    + _power_entries("SOFIA VALENTINA HERRERA CRUZ", LITIGATION_AND_ADMINISTRATION, 2),
                    2,
                ),
                "capital_social_actual": _scalar(
                    {"monto_numerico": 500000.0, "moneda": "MXN", "monto_en_letra": "QUINIENTOS MIL PESOS 00/100 MONEDA NACIONAL", "fecha_acto_vigente": "2021-03-15", "escritura_origen": "45,120"},
                    1,
                ),
            },
            "revoked_representatives": [],
            "expected_human_review_fields": [],
            "amendment_history": _events((incorporation, "2021-03-15", "45,120", 1, None, 500000.0)),
        }
    if acta_id == "acta_b":
        return {
            "fields": {
                "razon_social": _scalar("COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. DE C.V.", 1),
                "fecha_constitucion": _scalar("2020-07-22", 1),
                "numero_escritura": _scalar("12,890", 1),
                "notario": _scalar(
                    {"nombre": "FERNANDO GARZA TREVINO", "numero_notaria": 42, "entidad_federativa": "Nuevo Leon"},
                    1,
                ),
                "socios": _scalar(
                    [
                        {"nombre": "JORGE ALBERTO SALINAS MORA", "rfc": None, "acciones": 150, "porcentaje": 60.0, "page": 2},
                        {"nombre": "PATRICIA EUGENIA LOZANO VELA", "rfc": None, "acciones": 100, "porcentaje": 40.0, "page": 2},
                    ],
                    2,
                ),
                "representantes_legales": _scalar(
                    [
                        {"nombre": "JORGE ALBERTO SALINAS MORA", "nombre_normalizado": "JORGE ALBERTO SALINAS MORA", "cargo": "Gerente General", "fecha_nombramiento": "2020-07-22", "page": 2},
                    ],
                    2,
                ),
                "poderes": _scalar(
                    _power_entries("JORGE ALBERTO SALINAS MORA", LITIGATION_AND_ADMINISTRATION, 2),
                    2,
                ),
                "capital_social_actual": _scalar(
                    {"monto_numerico": 250000.0, "moneda": "MXN", "monto_en_letra": "DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.", "fecha_acto_vigente": "2020-07-22", "escritura_origen": "12,890"},
                    2,
                ),
            },
            "revoked_representatives": [],
            "expected_human_review_fields": ["capital_social_actual"],
            "amendment_history": _events((incorporation, "2020-07-22", "12,890", 1, None, 250000.0)),
        }
    if acta_id == "acta_c":
        return {
            "fields": {
                "razon_social": _scalar("LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V.", 1),
                "fecha_constitucion": _scalar("2019-05-10", 1),
                "numero_escritura": _scalar("30,100", 1),
                "notario": _scalar(
                    {"nombre": "ARTURO VILLASEÑOR ORTIZ", "numero_notaria": 15, "entidad_federativa": "Jalisco"},
                    1,
                ),
                "socios": _scalar(
                    [
                        {"nombre": "GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V.", "rfc": None, "acciones": 800, "porcentaje": 80.0, "page": 1},
                        {"nombre": "MARIA ELENA TORRES VEGA", "rfc": None, "acciones": 200, "porcentaje": 20.0, "page": 1},
                    ],
                    1,
                ),
                "representantes_legales": _scalar(
                    [
                        {"nombre": "MARIA ELENA TORRES VEGA", "nombre_normalizado": "MARIA ELENA TORRES VEGA", "cargo": "Apoderada Legal", "fecha_nombramiento": "2019-05-10", "page": 2},
                        {"nombre": "ING. ALEJANDRO GÓMEZ RUIZ", "nombre_normalizado": "ALEJANDRO GOMEZ RUIZ", "cargo": "Administrador Unico", "fecha_nombramiento": "2024-08-20", "page": 4},
                    ],
                    2,
                ),
                "poderes": _scalar(
                    _power_entries("MARIA ELENA TORRES VEGA", LITIGATION_AND_ADMINISTRATION, 2)
                    + _power_entries("ALEJANDRO GOMEZ RUIZ", ALL_POWERS, 4),
                    2,
                ),
                "capital_social_actual": _scalar(
                    {"monto_numerico": 1000000.0, "moneda": "MXN", "monto_en_letra": "UN MILLON DE PESOS 00/100 MONEDA NACIONAL", "fecha_acto_vigente": "2024-08-20", "escritura_origen": "58,400"},
                    3,
                ),
            },
            "revoked_representatives": ["JUAN CARLOS PEREZ LOPEZ"],
            "expected_human_review_fields": [],
            "amendment_history": _events(
                (incorporation, "2019-05-10", "30,100", 1, None, 100000.0),
                (AmendmentType.CAPITAL_INCREASE.value, "2024-08-20", "58,400", 3, None, 1000000.0),
                (AmendmentType.REVOCATION.value, "2024-08-20", "58,400", 3, "SR. JUAN C. PEREZ LOPEZ", None),
                (AmendmentType.GRANT_OF_POWERS.value, "2024-08-20", "58,400", 4, "ING. ALEJANDRO GÓMEZ RUIZ", None),
            ),
        }
    raise KeyError(f"unknown acta_id: {acta_id}")
