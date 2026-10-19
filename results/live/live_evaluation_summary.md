# Evaluacion en vivo del pipeline

Generado: 2026-10-19T05:22:34Z. Corridas: 1 (solicitadas: 1). Documentos evaluados: 3 (fallidos: 0).

Cada corrida copia `data/samples` y la verdad de referencia a un directorio temporal con cache vacio y procesa todo en vivo: RapidOCR local, extraccion y relectura VLM con la Responses API y verificacion con el predicate de la Decisions API.

## Entorno

| Parametro | Valor |
|---|---|
| Modelo de extraccion | gpt-6-luna |
| Modelo de vision | gpt-6-luna |
| Modelo Decisions | gpt-6-luna |
| SDK openai | 3.33.0 |
| rapidocr-onnxruntime | 1.4.4 |
| Python | 3.12.15 |
| Plataforma | macOS-26.6-arm64-arm-64bit |

## Puntaje final por acta

| Acta | Corridas OK | Media | Min | Max | Desv. estandar | Falsos positivos criticos |
|---|---|---|---|---|---|---|
| acta_a | 1/1 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| acta_b | 1/1 | 0.800 | 0.800 | 0.800 | 0.000 | 0 |
| acta_c | 1/1 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |

Global por documento: media 0.933, min 0.800, max 1.000, desv. estandar 0.116. Puntaje global por corrida: media 0.933, min 0.933, max 0.933. Falsos positivos criticos totales: 0. Tasa de acierto de evidencia media: 1.000.

## Puntaje por campo

### acta_a

| Campo | Media | Min | Max | Tasa de aprobacion |
|---|---|---|---|---|
| fecha_constitucion | 1.000 | 1.000 | 1.000 | 1.000 |
| numero_escritura | 1.000 | 1.000 | 1.000 | 1.000 |
| capital_social_actual | 1.000 | 1.000 | 1.000 | 1.000 |
| razon_social | 1.000 | 1.000 | 1.000 | 1.000 |
| notario | 1.000 | 1.000 | 1.000 | 1.000 |
| socios | 1.000 | 1.000 | 1.000 | 1.000 |
| representantes_legales | 1.000 | 1.000 | 1.000 | 1.000 |
| poderes | 1.000 | 1.000 | 1.000 | 1.000 |

### acta_b

| Campo | Media | Min | Max | Tasa de aprobacion |
|---|---|---|---|---|
| fecha_constitucion | 1.000 | 1.000 | 1.000 | 1.000 |
| numero_escritura | 1.000 | 1.000 | 1.000 | 1.000 |
| capital_social_actual | 0.000 | 0.000 | 0.000 | 0.000 |
| razon_social | 1.000 | 1.000 | 1.000 | 1.000 |
| notario | 1.000 | 1.000 | 1.000 | 1.000 |
| socios | 1.000 | 1.000 | 1.000 | 1.000 |
| representantes_legales | 1.000 | 1.000 | 1.000 | 1.000 |
| poderes | 1.000 | 1.000 | 1.000 | 1.000 |

### acta_c

| Campo | Media | Min | Max | Tasa de aprobacion |
|---|---|---|---|---|
| fecha_constitucion | 1.000 | 1.000 | 1.000 | 1.000 |
| numero_escritura | 1.000 | 1.000 | 1.000 | 1.000 |
| capital_social_actual | 1.000 | 1.000 | 1.000 | 1.000 |
| razon_social | 1.000 | 1.000 | 1.000 | 1.000 |
| notario | 1.000 | 1.000 | 1.000 | 1.000 |
| socios | 1.000 | 1.000 | 1.000 | 1.000 |
| representantes_legales | 1.000 | 1.000 | 1.000 | 1.000 |
| poderes | 1.000 | 1.000 | 1.000 | 1.000 |

## Enrutamiento a revision humana

Sobre-revision: campo marcado que no se esperaba. Sub-revision: campo esperado que no se marco (el caso peligroso). Precision = VP / (VP + sobre-revision); recall = VP / (VP + sub-revision). Solo documentos procesados sin error.

| Acta | Esperados | VP | Sobre-revision | Sub-revision | Precision | Recall | Coincidencia exacta | Campos sobre-revisados | Campos sub-revisados |
|---|---|---|---|---|---|---|---|---|---|
| acta_a | ninguno | 0 | 0 | 0 | n/d | n/d | 1/1 | - | - |
| acta_b | capital_social_actual | 1 | 0 | 0 | 1.000 | 1.000 | 1/1 | - | - |
| acta_c | ninguno | 0 | 0 | 0 | n/d | n/d | 1/1 | - | - |
| total | - | 1 | 0 | 0 | 1.000 | 1.000 | 3/3 | - | - |

## Latencia por documento (ms)

| Acta | Media | p50 | Max |
|---|---|---|---|
| acta_a | 35310.6 | 35310.6 | 35310.6 |
| acta_b | 85979.6 | 85979.6 | 85979.6 |
| acta_c | 48691.7 | 48691.7 | 48691.7 |
| global | 56660.6 | 48691.7 | 85979.6 |

## Tokens por etapa (media por documento)

| Acta | Etapa | Llamadas | Tokens de entrada | Tokens de salida |
|---|---|---|---|---|
| acta_a | extraction | 1.00 | 5373.0 | 4167.0 |
| acta_a | vlm_reread | 0.00 | 0.0 | 0.0 |
| acta_a | decisions_predicate | 0.00 | 0.0 | 0.0 |
| acta_b | extraction | 2.00 | 9830.0 | 6367.0 |
| acta_b | vlm_reread | 2.00 | 5862.0 | 1966.0 |
| acta_b | decisions_predicate | 8.00 | 7268.0 | 0.0 |
| acta_c | extraction | 1.00 | 5780.0 | 5582.0 |
| acta_c | vlm_reread | 0.00 | 0.0 | 0.0 |
| acta_c | decisions_predicate | 0.00 | 0.0 | 0.0 |

## Costo

- Decisions API: 7268 tokens de entrada en total; a USD 0.10 por millon, costo USD 0.000727.
- Responses API (gpt-6-luna, extraccion y relectura VLM): 26845 tokens de entrada y 18082 de salida en total; no se asume precio.

## Errores

- Ninguno.
