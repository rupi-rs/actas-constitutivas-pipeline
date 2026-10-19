"""Build the two-page architecture deliverable ``docs/arquitectura.pdf``.

Page 1 draws the high-level pipeline diagram with ReportLab shapes and
arrows. Page 2 holds the one-page architecture decisions text. The output is
byte-reproducible because the canvas is created with ``invariant=1``.

Notes
-----
Run from the repository root::

    uv run python docs/build_architecture_pdf.py

The document text is Spanish by design; it is rendered with the standard
Helvetica family, whose WinAnsi encoding covers Spanish accented letters.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Paragraph

OUTPUT_PATH: Path = Path(__file__).resolve().parent / "arquitectura.pdf"
"""Destination of the generated PDF."""

INK: Color = HexColor("#1F2933")
MUTED: Color = HexColor("#52606D")
LINE: Color = HexColor("#3E4C59")
FILL_RULE: Color = HexColor("#E4ECF5")
FILL_AI: Color = HexColor("#FCEFD9")
FILL_HUMAN: Color = HexColor("#F7E1E1")
FILL_STORE: Color = HexColor("#ECEFF1")

BOX_W: float = 110.0
BOX_H: float = 60.0
COLUMNS: tuple[float, ...] = (30.0, 154.0, 278.0, 402.0, 526.0, 650.0)
ROWS: tuple[float, ...] = (470.0, 385.0, 300.0, 215.0)


@dataclass(frozen=True)
class Box:
    """Rectangular diagram node.

    Parameters
    ----------
    x : float
        Left edge in points.
    y : float
        Bottom edge in points.
    title : str
        Bold heading.
    body : str
        Wrapped description text.
    fill : Color
        Background color that encodes the component category.
    w : float, default BOX_W
        Width in points.
    h : float, default BOX_H
        Height in points.
    """

    x: float
    y: float
    title: str
    body: str
    fill: Color
    w: float = BOX_W
    h: float = BOX_H

    @property
    def cx(self) -> float:
        """float: Horizontal center."""
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        """float: Vertical center."""
        return self.y + self.h / 2.0

    @property
    def top(self) -> float:
        """float: Top edge."""
        return self.y + self.h

    @property
    def right(self) -> float:
        """float: Right edge."""
        return self.x + self.w


def node(col: int, row: int, title: str, body: str, fill: Color) -> Box:
    """Create a box placed on the diagram grid.

    Parameters
    ----------
    col : int
        Column index into ``COLUMNS``.
    row : int
        Row index into ``ROWS``.
    title : str
        Bold heading.
    body : str
        Description text.
    fill : Color
        Category color.

    Returns
    -------
    Box
        Positioned box.
    """
    return Box(COLUMNS[col], ROWS[row], title, body, fill)


def draw_box(c: Canvas, box: Box) -> None:
    """Draw a box with its title and wrapped body text.

    Parameters
    ----------
    c : Canvas
        Target canvas.
    box : Box
        Node to draw.
    """
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.setFillColor(box.fill)
    c.roundRect(box.x, box.y, box.w, box.h, 4, stroke=1, fill=1)
    c.setFillColor(INK)
    title_lines = simpleSplit(box.title, "Helvetica-Bold", 8, box.w - 10)
    body_lines = simpleSplit(box.body, "Helvetica", 7, box.w - 10)
    y = box.top - 12
    c.setFont("Helvetica-Bold", 8)
    for line in title_lines:
        c.drawString(box.x + 5, y, line)
        y -= 9.5
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    for line in body_lines:
        c.drawString(box.x + 5, y, line)
        y -= 8.4


def _arrow_head(c: Canvas, x: float, y: float, dx: float, dy: float) -> None:
    """Draw a filled arrow head pointing along an axis-aligned direction.

    Parameters
    ----------
    c : Canvas
        Target canvas.
    x, y : float
        Tip coordinates.
    dx, dy : float
        Unit direction of the final segment (one of them is zero).
    """
    size = 4.5
    px, py = -dy, dx
    path = c.beginPath()
    path.moveTo(x, y)
    path.lineTo(x - dx * size * 1.6 + px * size, y - dy * size * 1.6 + py * size)
    path.lineTo(x - dx * size * 1.6 - px * size, y - dy * size * 1.6 - py * size)
    path.close()
    c.setFillColor(LINE)
    c.drawPath(path, stroke=0, fill=1)


def polyline_arrow(c: Canvas, points: list[tuple[float, float]], *, dashed: bool = False) -> None:
    """Draw an orthogonal or straight polyline ending in an arrow head.

    Parameters
    ----------
    c : Canvas
        Target canvas.
    points : list of tuple of float
        Vertices from start to tip; the last segment sets the head direction.
    dashed : bool, default False
        Draw a dashed line (used for the cache-hit shortcut).
    """
    c.setStrokeColor(LINE)
    c.setLineWidth(0.9)
    if dashed:
        c.setDash(3, 2)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()
    (x0, y0), (x1, y1) = points[-2], points[-1]
    length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6)
    _arrow_head(c, x1, y1, (x1 - x0) / length, (y1 - y0) / length)


def label(c: Canvas, x: float, y: float, text: str, *, align: str = "left") -> None:
    """Write a small italic connector label.

    Parameters
    ----------
    c : Canvas
        Target canvas.
    x, y : float
        Anchor position.
    text : str
        Label text.
    align : str, default "left"
        ``"left"``, ``"center"`` or ``"right"``.
    """
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 6.5)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def draw_legend(c: Canvas, x: float, y: float) -> None:
    """Draw the color legend for component categories.

    Parameters
    ----------
    c : Canvas
        Target canvas.
    x, y : float
        Baseline of the legend row.
    """
    entries = (
        (FILL_RULE, "Determinístico"),
        (FILL_AI, "Modelo de IA (acotado)"),
        (FILL_HUMAN, "Revisión humana"),
        (FILL_STORE, "Persistencia y control"),
    )
    c.setFont("Helvetica", 7)
    for fill, text in entries:
        c.setFillColor(fill)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)
        c.rect(x, y - 1, 9, 7, stroke=1, fill=1)
        c.setFillColor(INK)
        c.drawString(x + 13, y, text)
        x += 13 + c.stringWidth(text, "Helvetica", 7) + 16


def draw_diagram_page(c: Canvas) -> None:
    """Draw page 1: the high-level architecture diagram.

    Parameters
    ----------
    c : Canvas
        Canvas positioned on a fresh landscape letter page.
    """
    width, height = landscape(letter)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 40, "Arquitectura del pipeline de extracción de actas constitutivas")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(MUTED)
    c.drawString(
        30,
        height - 54,
        "Workflow determinístico con pasos de IA acotados. Las páginas nativas omiten OCR, relectura VLM y Decisions.",
    )
    draw_legend(c, 30, height - 72)

    entrada = node(0, 0, "Entrada", "PDF nativo, PDF escaneado o imagen PNG/JPG", FILL_RULE)
    ingesta = node(0, 1, "Ingesta", "SHA-256 sobre los bytes originales; job_id", FILL_RULE)
    cache = node(0, 2, "Caché por SHA-256", "Fallo: va al inspector. Acierto: omite OCR, VLM y LLM", FILL_STORE)
    inspector = node(1, 2, "Inspector por página", "PyMuPDF: >= 50 caracteres limpios es nativa; < 50 o imagen va a OCR", FILL_RULE)
    nativo = node(2, 1, "Nivel 0: texto nativo", "Capa de texto del PDF; confianza 1.0; costo 0", FILL_RULE)
    ocr = node(2, 2, "Nivel 1: OCR con bbox", "Render 200 dpi; confianza y bbox por región; >= 0.85 pasa directo", FILL_RULE)
    vlm = node(2, 3, "Nivel 2: relectura VLM", "Recorte físico por bbox; gpt-6-luna visión; esquema {reading, legible}", FILL_AI)
    texto = node(3, 2, "Texto por página", "Corregido con la lectura VLM; se conservan ambas lecturas", FILL_RULE)
    extraccion = node(3, 0, "Extracción LLM", "Responses API, JSON schema estricto; páginas en XML <page>", FILL_AI)
    resolver = node(4, 0, "Interpretación", "Resolución temporal; revocaciones con RapidFuzz >= 85 y guarda de apellidos", FILL_RULE)
    validacion = node(4, 1, "Validación determinística", "Grounding de cita, valor en cita, fechas, monto en letra, spec del tipo", FILL_RULE)
    decisions = node(4, 2, "Nivel 2: Decisions API", "predicate sobre el recorte: |2p-1| frente a 0.80; score de legibilidad e inyección", FILL_AI)
    ruteo = node(5, 2, "Enrutamiento", "Umbrales por campo; motivo de revisión explícito", FILL_RULE)
    revision = node(5, 3, "Revisión humana", "Recorte PNG, cita, página, ambas lecturas y motivo; la decisión vuelve al JSON", FILL_HUMAN)
    salida = node(5, 1, "JSON versionado", "Evidencia por campo; audit_metadata con schema_version y model_version", FILL_STORE)
    consumidores = node(5, 0, "Consumidores", "Alta de cuentas, representantes, contratos, expediente", FILL_STORE)

    for box in (
        entrada, ingesta, cache, inspector, nativo, ocr, vlm, texto, extraccion,
        resolver, validacion, decisions, ruteo, revision, salida, consumidores,
    ):
        draw_box(c, box)

    polyline_arrow(c, [(entrada.cx, entrada.y), (ingesta.cx, ingesta.top)])
    polyline_arrow(c, [(ingesta.cx, ingesta.y), (cache.cx, cache.top)])
    polyline_arrow(c, [(cache.right, cache.cy), (inspector.x, inspector.cy)])

    mid_x = inspector.right + 7
    polyline_arrow(c, [(inspector.right, inspector.cy + 12), (mid_x, inspector.cy + 12), (mid_x, nativo.cy), (nativo.x, nativo.cy)])
    label(c, inspector.cx, inspector.top + 3, "nativa >= 50", align="center")
    polyline_arrow(c, [(inspector.right, inspector.cy - 8), (ocr.x, inspector.cy - 8)])

    polyline_arrow(c, [(ocr.cx, ocr.y), (vlm.cx, vlm.top)])
    label(c, ocr.cx + 3, ocr.y - 10, "regiones < 0.85")
    polyline_arrow(c, [(ocr.right, ocr.cy), (texto.x, texto.cy)])
    polyline_arrow(c, [(vlm.right, vlm.cy), (texto.cx, vlm.cy), (texto.cx, texto.y)])
    polyline_arrow(c, [(nativo.right, nativo.cy), (texto.cx, nativo.cy), (texto.cx, texto.top)])
    polyline_arrow(c, [(texto.cx + 30, texto.top), (texto.cx + 30, extraccion.y)])
    label(c, texto.cx + 27, (texto.top + extraccion.y) / 2.0 + 20, "páginas XML", align="right")

    polyline_arrow(c, [(extraccion.right, extraccion.cy), (resolver.x, extraccion.cy)])
    polyline_arrow(c, [(resolver.cx, resolver.y), (validacion.cx, validacion.top)])
    polyline_arrow(c, [(validacion.cx, validacion.y), (decisions.cx, decisions.top)])
    polyline_arrow(c, [(decisions.right, decisions.cy), (ruteo.x, decisions.cy)])
    polyline_arrow(c, [(ruteo.cx, ruteo.top), (salida.cx, salida.y)])
    label(c, ruteo.cx + 3, ruteo.top + 9, "sin dudas")
    polyline_arrow(c, [(ruteo.cx, ruteo.y), (revision.cx, revision.top)])
    label(c, ruteo.cx + 3, ruteo.y - 10, "campo dudoso")
    side_x = salida.right + 12
    polyline_arrow(c, [(revision.right, revision.cy), (side_x, revision.cy), (side_x, salida.cy), (salida.right, salida.cy)])
    polyline_arrow(c, [(salida.cx, salida.top), (consumidores.cx, consumidores.y)])

    hit_y = vlm.y - 14
    polyline_arrow(
        c,
        [(cache.cx, cache.y), (cache.cx, hit_y), (resolver.x - 7, hit_y), (resolver.x - 7, resolver.cy - 14), (resolver.x, resolver.cy - 14)],
        dashed=True,
    )
    label(c, cache.cx + 4, hit_y + 3, "acierto: extracción cruda servida desde caché")

    band_y = 112.0
    band_h = 58.0
    bands = (
        Box(30.0, band_y, w=330.0, h=band_h, title="Almacenamiento por capas",
            body="Binario original (SHA-256); texto por página y fuente; regiones OCR con bbox y confianza; "
            "lecturas OCR y VLM; extracción cruda previa a la resolución (caché); JSON final con "
            "page_number, source_quote, confidence_source y motivo de revisión.", fill=FILL_STORE),
        Box(374.0, band_y, w=180.0, h=band_h, title="Bitácora de auditoría",
            body="JSON Lines con job_id, doc_sha256, stage, event, duration_ms y details; "
            "PIIRedactionFilter enmascara RFC, CURP y correo.", fill=FILL_STORE),
        Box(568.0, band_y, w=192.0, h=band_h, title="Evaluación y monitoreo",
            body="Métricas por campo contra ground truth, Evidence Hit Rate, falsos positivos críticos "
            "(revocados listados como activos) y tasa de revisión.", fill=FILL_STORE),
    )
    for band in bands:
        draw_box(c, band)
    label(c, 30, band_y + band_h + 6, "Capa transversal: todas las etapas persisten artefactos y registran eventos.")

    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(
        30,
        84,
        "Producción: recepción síncrona (hash, caché, HTTP 202 con job_id); OCR, VLM, LLM, verificación y revisión "
        "asíncronas en cola con reintentos acotados y DLQ.",
    )
    c.drawString(
        30,
        74,
        "Prototipo: CLI por lotes (actas-cli). Reintentos tenacity solo para errores transitorios; rechazos y salidas "
        "fuera de esquema van a revisión.",
    )
    c.drawRightString(width - 30, 30, "Página 1 de 2")


DECISIONS: tuple[tuple[str, str], ...] = (
    (
        "1. PDF nativo frente a escaneado.",
        "Se decide por página: PyMuPDF extrae la capa de texto y se cuentan caracteres alfanuméricos limpios. "
        "Con 50 o más la página es nativa (confianza 1.0); con menos se renderiza a 200 dpi y entra a OCR. "
        "Un documento puede ser híbrido. Una imagen PNG/JPG se trata como página escaneada. Solo cambia la fuente "
        "del texto y su confianza; extracción, resolución y validación son las mismas.",
    ),
    (
        "2. Nativo, OCR, multimodal o LLM de texto.",
        "Nativo siempre que haya capa de texto: exacto, gratuito y determinístico. OCR local con bbox (Nivel 1) para páginas "
        "sin texto, con confianza y bbox por región. Modelo multimodal solo sobre recortes con confianza OCR menor "
        "a 0.85 (relectura VLM con esquema {reading, legible}), nunca sobre el documento completo. LLM de texto para "
        "la extracción a esquema estricto sobre el texto corregido. Decisions (predicate) verifica el valor final "
        "sobre el mismo recorte y devuelve una probabilidad p que se mapea a |2p - 1|.",
    ),
    (
        "3. Qué se conserva.",
        "Original direccionado por SHA-256, texto por página con su fuente, regiones OCR con bbox y confianza, "
        "lecturas OCR y VLM, extracción cruda previa a la resolución (se cachea para re-ejecutar siempre la lógica "
        "determinística), JSON final con page_number, source_quote, confidence_source, bbox y motivo de revisión, "
        "audit_metadata (schema_version, model_version), prompt_sha256 en la entrada de caché y bitácora JSONL con PII "
        "enmascarada.",
    ),
    (
        "4. Síncrono y asíncrono.",
        "Síncrono: recepción, hash, consulta de caché y respuesta HTTP 202 con job_id (o el resultado si ya existe). "
        "Asíncrono: OCR, VLM, extracción, verificación y revisión humana, en cola con reintentos acotados y DLQ.",
    ),
    (
        "5. Reglas, ML y GenAI.",
        "Reglas para todo lo verificable: enrutamiento, grounding, valor en cita, número frente a letra, fechas, "
        "resolución temporal y revocaciones (RapidFuzz >= 85 con guarda de apellidos), umbrales. ML clásico para OCR "
        "y clasificación documental. GenAI solo para lenguaje no estructurado: extracción, relectura visual y "
        "verificación por predicado.",
    ),
    (
        "6. Agente o workflow.",
        "Workflow determinístico con pasos de IA acotados. El orden de etapas, los umbrales y la decisión de revisión "
        "los fija el código; cada llamada a un modelo tiene entrada, esquema y uso definidos. Un agente autónomo "
        "añade variabilidad y superficie de inyección difícil de auditar, sin beneficio para un documento de "
        "estructura conocida.",
    ),
    (
        "7. Revisión humana.",
        "Decisions calibrado menor a 0.80; VLM declara la región ilegible; OCR y VLM discrepan sin confirmación alta "
        "de Decisions; falla el grounding (confianza topada en 0.40); el valor no está en su cita; incoherencia "
        "temporal; revocación ambigua (similitud 75 a 85 o apellidos distintos); campo requerido ausente; verificador "
        "no disponible o con rechazo; legibilidad o riesgo de inyección fuera de umbral (preguntas score). El revisor ve "
        "el recorte guardado en outputs/crops/{acta_id}_p{page}_{field}_bbox.png, la cita, ambas lecturas y el motivo.",
    ),
    (
        "Por qué la API Decisions.",
        "POST /v1/decisions devuelve por pregunta una probability (predicate) o un score con confidence (score, choice); "
        "es la primitiva diseñada para esta tarea y solo factura entrada (USD 0.10 por millón). Una confianza "
        "verbalizada por un modelo generativo no se usa para decidir, y no se usan modelos de razonamiento extendido. "
        "Extracción y VLM pueden ir a un servidor on-prem compatible con OpenAI (OPENAI_BASE_URL); Decisions requiere "
        "OpenAI Cloud; sin conexión se usa la caché determinística.",
    ),
    (
        "Reutilización y evolución.",
        "Se extrae una vez por documento; los consumidores (alta de cuentas, validación de representantes, firma de "
        "contratos, expediente) leen el mismo JSON versionado y aplican su política sobre status, evidencia y "
        "revisión. Un tipo documental nuevo se agrega registrando un DocumentTypeSpec en DocumentRegistry, sin tocar "
        "el orquestador.",
    ),
    (
        "Fallos, reintentos e idempotencia.",
        "Reintentos con tenacity (tres intentos y backoff exponencial) solo para errores transitorios; el SDK usa max_retries=0. "
        "Rechazos, salidas fuera de esquema o verificador no disponible van a revisión con motivo. Un resultado con "
        "fallas de grounding no se cachea. Un cambio de prompt (prompt_sha256) invalida entradas vivas.",
    ),
    (
        "Privacidad, observabilidad y costo.",
        "PIIRedactionFilter en consola y bitácora; extracción con store=False. Costo variable concentrado en páginas "
        "escaneadas y extracción LLM: una verificación Decisions usó 867 tokens de entrada (USD 0.10 por millón); la "
        "extracción del Acta A usó 5,087 tokens de entrada y 5,101 de salida. Escalado horizontal con workers sin "
        "estado detrás de la cola.",
    ),
)
"""Decision headings and texts rendered on page 2."""


def draw_decisions_page(c: Canvas) -> None:
    """Draw page 2: the one-page architecture decisions text.

    Parameters
    ----------
    c : Canvas
        Canvas positioned on a fresh portrait letter page.

    Raises
    ------
    RuntimeError
        If the decisions text does not fit on a single page.
    """
    width, height = letter
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK, spaceAfter=4)
    intro = ParagraphStyle("intro", fontName="Helvetica", fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=6)
    body = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.6, leading=12.8, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7
    )
    story = [
        Paragraph("Decisiones de arquitectura", title),
        Paragraph("Respuestas a las preguntas de diseño y criterios transversales. Detalle en docs/ARCHITECTURE_AND_DESIGN.md.", intro),
    ]
    story.extend(Paragraph(f"<b>{heading}</b> {text}", body) for heading, text in DECISIONS)
    frame = Frame(48, 48, width - 96, height - 96, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame.addFromList(story, c)
    if story:
        raise RuntimeError("decisions text does not fit on one page")
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawRightString(width - 48, 30, "Página 2 de 2")


def build(output_path: Path = OUTPUT_PATH) -> Path:
    """Render the two-page architecture PDF.

    Parameters
    ----------
    output_path : Path, default OUTPUT_PATH
        Destination file.

    Returns
    -------
    Path
        Path of the written PDF.
    """
    c = Canvas(str(output_path), pagesize=landscape(letter), invariant=1)
    c.setTitle("Arquitectura del pipeline de extracción de actas constitutivas")
    c.setSubject("Diagrama de alto nivel y decisiones de arquitectura")
    draw_diagram_page(c)
    c.showPage()
    c.setPageSize(letter)
    draw_decisions_page(c)
    c.showPage()
    c.save()
    return output_path


def main() -> int:
    """Command-line entry point.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    build()
    sys.stdout.write(f"[OK] architecture PDF written to {OUTPUT_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
