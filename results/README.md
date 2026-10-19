# Resultados

Salidas de una ejecución determinística de `actas-cli demo` (sin credenciales; OCR, relectura VLM, extracción y verificación servidos desde la caché preverificada por SHA-256), más los reportes de `actas-cli test-decisions` en modo simulado y en vivo. Son copias de `outputs/`, que el pipeline regenera en cada ejecución.

| Archivo | Contenido |
|---------|-----------|
| `acta_a.json` | Salida estructurada del Acta A (constitución nativa, 2 páginas) |
| `acta_b.json` | Salida del Acta B (página 1 nativa, página 2 escaneada con sello sobre el capital); incluye los campos de relectura en la evidencia (`ocr_reading`, `vlm_reading`, `vlm_legible`, `verification_probability`, `crop_path`, `page_legibility_score`) |
| `acta_c.json` | Salida del Acta C (constitución 2019 y asamblea 2024, 4 páginas) |
| `evaluation_report.json` | Comparación contra `data/ground_truth/ground_truth.json`: métrica, calificación y detalle por campo, Evidence Hit Rate, falsos positivos críticos y coincidencia del enrutamiento a revisión |
| `decisions_verification_report_simulated.json` | Reporte de `actas-cli test-decisions --no-live`: cuatro casos con respuestas simuladas (`simulated: true`): predicado sobre texto limpio, predicado sobre el recorte sellado del Acta B, predicado de contradicción del representante revocado del Acta C y puntaje de riesgo de inyección |
| `decisions_verification_report_live.json` | Reporte de `actas-cli test-decisions --live` (2026-10-19, `gpt-6-luna`, API Decisions real): mismos cuatro casos con respuestas reales, 1,891 tokens de entrada en total |
| `crops/acta_b_p2_capital_social_actual_bbox.png` | Recorte PNG (200 dpi) de la región del capital del Acta B bajo el sello, el mismo que se envía a la relectura VLM y al predicado de Decisions |

Cada JSON de acta contiene los ocho campos del contrato (`razon_social`, `fecha_constitucion`, `numero_escritura`, `notario`, `socios`, `representantes_legales`, `poderes`, `capital_social_actual`), cada uno con valor, estado, evidencia (página, cita literal, confianza y fuente) y bandera de revisión, más `amendment_history` y `audit_metadata` (incluye `security_flags`). El esquema se explica en [docs/SCHEMA.md](../docs/SCHEMA.md).

## Comparación con el ground truth

| Documento | Campos aprobados | Final | Evidence Hit Rate | Falsos positivos críticos | Revisión esperada / marcada |
|-----------|------------------|-------|-------------------|---------------------------|-----------------------------|
| Acta A | 8 de 8 | 1.000 | 100.0% | 0 | ninguna / ninguna |
| Acta B | 8 de 8 | 1.000 | 100.0% | 0 | `capital_social_actual` / `capital_social_actual` |
| Acta C | 8 de 8 | 1.000 | 100.0% | 0 | ninguna / ninguna |

Calificación global 1.000, sin penalización global.

- Acta B: `capital_social_actual` se extrae (250,000.00 MXN) y se envía a revisión con el motivo `Low OCR confidence (0.72) and Decisions predicate confidence (0.56 < 0.80) due to overlapping notary seal on page 2 bbox [72.0, 410.0, 520.0, 455.0].` En esta ejecución determinística `p = 0.78` es la simulación del escenario especificado.
- Acta C: el capital vigente es 1,000,000.00 (escritura 58,400 del 2024-08-20); `SR. JUAN C. PEREZ LOPEZ` se cruza con `LIC. JUAN CARLOS PÉREZ LÓPEZ` (similitud 89.41) y se elimina de representantes y poderes; la evidencia del nombramiento y de la revocación queda en `amendment_history`.

## Reportes de Decisions: simulado frente a en vivo

| Caso | Simulado | En vivo (2026-10-19) | Veredicto |
|------|----------|----------------------|-----------|
| Predicado, texto limpio del capital | `p = 0.98` | `p = 1.00` | CONFIRMED |
| Predicado, recorte sellado del Acta B | `p = 0.78`, calibrada 0.56 | `p = 0.48`, `abs(2p - 1) = 0.04` | REVIEW |
| Predicado, representante revocado del Acta C | `p = 0.03` | `p = 0.00` | REJECTED |
| Puntaje de riesgo de inyección | normalizado 0.9367 | normalizado 0.997 | FLAGGED |

Los valores simulados no son mediciones; el reporte en vivo es la medición real de la API.

Esta ejecución determinística valida la lógica del pipeline sobre una extracción conocida, no la calidad de los modelos. Los resultados de las ejecuciones en vivo con `gpt-6-luna` (extracción de Acta A y Acta B con calificación 1.000; relectura VLM del recorte sellado con `legible = false`; Acta B completa con RapidOCR real y legibilidad 0.33) se documentan en [docs/EVALUACION.md](../docs/EVALUACION.md) y en [docs/ARCHITECTURE_AND_DESIGN.md](../docs/ARCHITECTURE_AND_DESIGN.md), sección 12; esas ejecuciones no están como archivos en esta carpeta.

## Regenerar

```bash
uv sync
uv run actas-cli demo
cp outputs/acta_a.json outputs/acta_b.json outputs/acta_c.json outputs/evaluation_report.json results/
mkdir -p results/crops && cp outputs/crops/acta_b_p2_capital_social_actual_bbox.png results/crops/
uv run actas-cli test-decisions            # simulado
cp outputs/decisions_verification_report.json results/decisions_verification_report_simulated.json
```

El reporte en vivo se genera con `uv run actas-cli test-decisions --live` (requiere `OPENAI_API_KEY`). Entre ejecuciones cambian `audit_metadata.job_id` y `audit_metadata.processing_time_ms`; el resto del contenido es idéntico para los mismos PDFs.
