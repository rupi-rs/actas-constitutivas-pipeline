"""Build the two-page architecture deliverable ``docs/arquitectura.pdf``.

Page 1 is the Mermaid diagram rendered to PDF (``architecture_diagram.pdf``,
produced from ``architecture.mmd`` with mermaid-cli in Docker). Page 2 holds
the one-page architecture decisions text, rendered with ReportLab and merged
after the diagram with PyMuPDF.

Notes
-----
Run from the repository root::

    uv run python docs/build_architecture_pdf.py

The document text is Spanish by design; it is rendered with the standard
Helvetica family, whose WinAnsi encoding covers Spanish accented letters.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Frame, Paragraph

DOCS_DIR: Path = Path(__file__).resolve().parent
"""Directory holding the diagram sources and outputs."""

DIAGRAM_PATH: Path = DOCS_DIR / "architecture_diagram.pdf"
"""Mermaid-rendered diagram (single page)."""

OUTPUT_PATH: Path = DOCS_DIR / "arquitectura.pdf"
"""Destination of the generated PDF."""

DIAGRAM_PAGE_WIDTH: float = 842.0
"""Width in points of the diagram page (A4 landscape width)."""

DIAGRAM_MARGIN: float = 18.0
"""Margin in points around the scaled diagram."""

INK: Color = HexColor("#1F2933")
MUTED: Color = HexColor("#52606D")

DECISIONS: tuple[tuple[str, str], ...] = (
    (
        "1. PDF nativo frente a escaneado.",
        "Se decide por página: PyMuPDF extrae la capa de texto y cuenta caracteres alfanuméricos limpios. Con 50 o más "
        "la página es nativa (Nivel 0: confianza 1.0, costo 0, sin OCR, VLM ni Decisions); con menos se renderiza a "
        "200 dpi y entra al OCR. Un documento puede ser híbrido. Las imágenes PNG/JPG se envuelven como páginas "
        "escaneadas y el SHA-256 se calcula sobre los bytes originales. Solo cambian la fuente del texto, su confianza "
        "y la posible escalación; extracción, interpretación y validación son las mismas.",
    ),
    (
        "2. Nativo, OCR, multimodal o LLM de texto.",
        "Nativo siempre que haya capa de texto: exacto, gratuito y determinístico. OCR local con bbox y confianza por "
        "línea (Nivel 1) para páginas escaneadas. Modelo multimodal solo sobre recortes físicos (+12 pt, 200 dpi) cuando "
        "una región tiene confianza OCR menor a 0.85, o siempre que un campo crítico (fecha, escritura, notario, "
        "capital, representantes, poderes) cae en una página escaneada; la relectura devuelve {reading, legible} y se "
        "conservan ambas lecturas. LLM de texto para la extracción a esquema estricto. La API Decisions (predicate) "
        "es el único verificador: evalúa el valor sobre el recorte y p se mapea a |2p-1| frente a 0.80.",
    ),
    (
        "3. Qué se conserva para auditoría.",
        "Original inmutable direccionado por SHA-256; extracción cruda en caché previa a la resolución (la lógica "
        "determinística se re-ejecuta siempre); JSON validado y versionado (schema_version, versiones de modelo, "
        "prompt_sha256) con page_number, source_quote, confidence_source, bbox y motivo de revisión por campo; recortes "
        "en outputs/crops/; lecturas OCR y VLM; y bitácora JSONL con PII enmascarada.",
    ),
    (
        "4. Síncrono y asíncrono.",
        "Síncrono: recepción, hash, consulta de caché y respuesta HTTP 202 con job_id (o el resultado si ya existe). "
        "Asíncrono: cola, workers y DLQ para OCR, VLM, extracción, verificación y revisión; carril rápido para "
        "páginas nativas y carril para páginas escaneadas. El prototipo corre por lotes desde la CLI.",
    ),
    (
        "5. Reglas, ML y GenAI.",
        "Reglas para todo lo verificable: enrutamiento, prefiltro de inyección por expresiones regulares sin API, "
        "grounding, valor en cita, número frente a letra, coherencia de fechas, resolución temporal y revocaciones "
        "(RapidFuzz >= 85 con guarda de apellidos). ML clásico para el OCR. GenAI solo para lenguaje o imagen no "
        "estructurados: extracción, relectura visual de recortes dudosos y verificación por predicado.",
    ),
    (
        "6. Agente o workflow.",
        "Workflow determinístico con puntos de IA acotados, no un agente. El orden de etapas, los umbrales y la "
        "decisión de revisión los fija el código; cada llamada a un modelo tiene entrada, esquema estricto y uso "
        "únicos, y el texto del documento se trata como dato no confiable. Un agente añadiría variabilidad y "
        "superficie de inyección difícil de auditar. Los ciclos son acotados: ante un campo crítico ausente en una "
        "página escaneada, una relectura VLM y a lo sumo una reextracción.",
    ),
    (
        "7. Revisión humana (Mesa de control).",
        "Por campo, cuando |2p-1| < 0.80; la región propia es legible = false; falla un chequeo determinístico "
        "(grounding, valor en cita, consistencia temporal, número frente a letra, spec del registro); falla el "
        "verificador o el VLM; o el prefiltro de inyección marcó la página. El revisor ve el recorte, la cita, la "
        "página, ambas lecturas y el motivo; el resto se aprueba automáticamente.",
    ),
    (
        "Modos de proveedor.",
        "Modo 1 (defecto): OpenAI Cloud, gpt-6-luna y /v1/decisions. Modo 2: vLLM/TGI en OpenShift vía "
        "OPENAI_BASE_URL para extracción y VLM; Decisions solo en la nube: 2a envía únicamente el recorte, 2b omite "
        "Decisions y manda a revisión humana. Modo 3: caché determinística sin conexión. El orquestador depende solo "
        "de interfaces (BaseLLMProvider, BaseOCRProvider, BaseVerifierProvider, BaseStorageRepository).",
    ),
    (
        "Reutilización y evolución.",
        "Se extrae una vez por documento; los consumidores (alta de cuentas, validación de representantes legales, "
        "firma y revisión de contratos, actualización de expediente) leen el mismo JSON y aplican su política sobre "
        "estado, evidencia y revisión. Un tipo documental nuevo se registra en DocumentRegistry sin tocar el "
        "orquestador. La evaluación contra ground truth mide aciertos, evidencia y falsos positivos críticos.",
    ),
)
"""Decision headings and texts rendered on page 2."""


def draw_decisions_page(c: Canvas) -> None:
    """Draw the one-page architecture decisions text.

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
        "body", fontName="Helvetica", fontSize=9.2, leading=12.0, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5
    )
    story = [
        Paragraph("Decisiones de arquitectura", title),
        Paragraph("Respuestas a las preguntas de diseño; el diagrama está en la página anterior. Detalle en docs/ARCHITECTURE_AND_DESIGN.md.", intro),
    ]
    story.extend(Paragraph(f"<b>{heading}</b> {text}", body) for heading, text in DECISIONS)
    frame = Frame(48, 48, width - 96, height - 96, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame.addFromList(story, c)
    if story:
        raise RuntimeError("decisions text does not fit on one page")


def render_decisions_pdf() -> bytes:
    """Render the decisions page to an in-memory PDF.

    Returns
    -------
    bytes
        One-page PDF (letter, portrait).
    """
    buffer = io.BytesIO()
    c = Canvas(buffer, pagesize=letter, invariant=1)
    draw_decisions_page(c)
    c.showPage()
    c.save()
    return buffer.getvalue()


def build(output_path: Path = OUTPUT_PATH, diagram_path: Path = DIAGRAM_PATH) -> Path:
    """Merge the Mermaid diagram page and the decisions page.

    Parameters
    ----------
    output_path : Path, default OUTPUT_PATH
        Destination file.
    diagram_path : Path, default DIAGRAM_PATH
        Mermaid-rendered single-page PDF.

    Returns
    -------
    Path
        Path of the written PDF.

    Raises
    ------
    FileNotFoundError
        If the rendered Mermaid PDF does not exist.
    """
    if not diagram_path.is_file():
        raise FileNotFoundError(f"missing rendered diagram: {diagram_path}")
    with pymupdf.open(str(diagram_path)) as source:
        src_rect = source[0].rect
        inner_w = DIAGRAM_PAGE_WIDTH - 2 * DIAGRAM_MARGIN
        scale = inner_w / src_rect.width
        page_h = src_rect.height * scale + 2 * DIAGRAM_MARGIN
        merged = pymupdf.open()
        page = merged.new_page(width=DIAGRAM_PAGE_WIDTH, height=page_h)
        target = pymupdf.Rect(DIAGRAM_MARGIN, DIAGRAM_MARGIN, DIAGRAM_PAGE_WIDTH - DIAGRAM_MARGIN, page_h - DIAGRAM_MARGIN)
        page.show_pdf_page(target, source, 0)
    with pymupdf.open("pdf", render_decisions_pdf()) as decisions:
        merged.insert_pdf(decisions)
    merged.set_metadata(
        {
            "title": "Arquitectura del pipeline de extracción de actas constitutivas",
            "subject": "Diagrama de alto nivel y decisiones de arquitectura",
        }
    )
    merged.save(str(output_path), garbage=4, deflate=True)
    merged.close()
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
