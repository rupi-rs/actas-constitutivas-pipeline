# Resultados

La ejecución determinística con caché es una prueba de regresión, no una medición de desempeño: la caché preverificada refleja el ground truth, por lo que calificarla es circular. El desempeño real proviene solo de `actas-cli evaluate-live --runs N` (RapidOCR real, `gpt-6-luna` y Decisions, raíces temporales aisladas, sin caché), cuyos reportes se guardan en `results/live/`.

## Resultados en vivo (corrida final)

Evaluación final totalmente en vivo con `actas-cli evaluate-live --runs 1` (2026-10-19, código final): las muestras se copian a un directorio aislado con caché vacío y se procesan con RapidOCR 1.4.4, gpt-6-luna (extracción y relectura VLM vía Responses API) y el predicate de la Decisions API. 3 documentos, 0 errores. Detalle en `results/live/live_evaluation_summary.md` y `results/live/live_evaluation_report.json`; salidas y bitácora en `results/live/runs/`.

| Acta | Puntaje final | Campos correctos | Falsos positivos críticos | Tasa de acierto de evidencia | Revisión humana (esperado / marcado) | Latencia |
|---|---|---|---|---|---|---|
| A (nativa) | 1.000 | 8/8 | 0 | 1.000 | ninguno / ninguno | 35.3 s |
| B (escaneada con sello) | 0.800 | 7/8 (capital `no_encontrado`) | 0 | 1.000 | capital / capital | 86.0 s |
| C (reforma y revocación) | 1.000 | 8/8 | 0 | 1.000 | ninguno / ninguno | 48.7 s |

Enrutamiento a revisión humana: precisión 1.000 y recall 1.000 (sin sobre-revisión ni sub-revisión). Puntaje global 0.933. El Acta B pierde el peso del capital (0.20) porque el monto "$250,000.00" está realmente cubierto por la tinta del sello: RapidOCR lo omite, la relectura VLM responde `legible=false` ("$ MIL PESOS") y, tras una única reextracción acotada, el campo queda `no_encontrado` con revisión humana ("Required field capital_social_actual not found after VLM re-read of scanned page(s) 2 and one re-extraction"). El sistema no adivina un monto que no se ve; aprobar un monto inventado sería el error grave y no ocurrió. Juan Carlos Pérez López quedó revocado. Prueba adicional en vivo con `process-file`: el Acta A convertida en dos imágenes escaneadas se extrajo completa y sin ningún campo en revisión.

Tokens y costo: las Actas A y C (nativas) usan una sola llamada de extracción y no llaman al VLM ni a Decisions. El Acta B usa 2 extracciones, 2 relecturas VLM y 8 predicates (un predicate por valor: capital, cada socio, representante y poder). Decisions: 7,268 tokens de entrada (USD 0.0007 a USD 0.10 por millón). Responses API: 26,845 tokens de entrada y 18,082 de salida; no se asume tarifa.

Historial (`results/live_history/three_runs_before_predicate_fix/`): 3 corridas con el código previo dieron 1.000 en las tres actas (el modelo sí completó el capital de B) pero con sobre-revisión de `socios` en 3/3 (precisión 0.500), causada por predicates de cita textual y por enviar la página OCR completa como contexto; ambos se corrigieron (predicates por valor sobre el recorte y contexto solo de la región). Alcance: documentos sintéticos; mide el funcionamiento con modelos reales, no la precisión esperada en producción.

## Regresión determinística y verificación de Decisions

Salidas de una ejecución determinística de `actas-cli demo` (sin credenciales; OCR, relectura VLM, extracción y verificación servidos desde la caché preverificada por SHA-256), más los reportes de `actas-cli test-decisions` en modo simulado y en vivo. Son copias de `outputs/`, que el pipeline regenera en cada ejecución.

| Archivo | Contenido |
|---------|-----------|
| `acta_a.json` | Salida estructurada del Acta A (constitución nativa, 2 páginas) |
| `acta_b.json` | Salida del Acta B (página 1 nativa, página 2 escaneada con sello sobre el capital); incluye los campos de relectura en la evidencia (`ocr_reading`, `vlm_reading`, `vlm_legible`, `verification_probability`, `crop_path`) |
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

Son la verificación independiente `actas-cli test-decisions`, que no forma parte del procesamiento de documentos.

| Caso | Simulado | En vivo (2026-10-19) | Veredicto |
|------|----------|----------------------|-----------|
| Predicado, texto limpio del capital | `p = 0.98` | `p = 1.00` | CONFIRMED |
| Predicado, recorte sellado del Acta B | `p = 0.78`, calibrada 0.56 | `p = 0.48`, `abs(2p - 1) = 0.04` | REVIEW |
| Predicado, representante revocado del Acta C | `p = 0.03` | `p = 0.00` | REJECTED |
| Puntaje de riesgo de inyección | normalizado 0.9367 | normalizado 0.997 | FLAGGED |

Los valores simulados no son mediciones; el reporte en vivo es la medición real de la API.

Esta ejecución determinística valida la lógica del pipeline sobre una extracción conocida, no la calidad de los modelos. Los sondeos exploratorios de una sola corrida con `gpt-6-luna` (extracción de Acta A y Acta B con calificación 1.000; relectura VLM del recorte sellado con `legible = false`; Acta B completa con RapidOCR real) se documentan en [docs/EVALUACION.md](../docs/EVALUACION.md) y en [docs/ARCHITECTURE_AND_DESIGN.md](../docs/ARCHITECTURE_AND_DESIGN.md), sección 12; esas ejecuciones no están como archivos en esta carpeta.

## Regenerar

```bash
uv sync
uv run actas-cli demo
cp outputs/acta_a.json outputs/acta_b.json outputs/acta_c.json outputs/evaluation_report.json results/
mkdir -p results/crops && cp outputs/crops/acta_b_p2_capital_social_actual_bbox.png results/crops/
uv run actas-cli test-decisions            # simulado
cp outputs/decisions_verification_report.json results/decisions_verification_report_simulated.json
```

La evaluación en vivo se genera con `uv run actas-cli evaluate-live --runs 3` (requiere `OPENAI_API_KEY` y `uv sync --extra ocr`). El reporte en vivo de Decisions se genera con `uv run actas-cli test-decisions --live` (requiere `OPENAI_API_KEY`). Entre ejecuciones cambian `audit_metadata.job_id` y `audit_metadata.processing_time_ms`; el resto del contenido es idéntico para los mismos PDFs.
