# Evaluacion en vivo del pipeline

Generado: 2026-10-19T04:49:02Z. Corridas: 3 (solicitadas: 3). Documentos evaluados: 9 (fallidos: 0).

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
| acta_a | 3/3 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| acta_b | 3/3 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |
| acta_c | 3/3 | 1.000 | 1.000 | 1.000 | 0.000 | 0 |

Global por documento: media 1.000, min 1.000, max 1.000, desv. estandar 0.000. Puntaje global por corrida: media 1.000, min 1.000, max 1.000. Falsos positivos criticos totales: 0. Tasa de acierto de evidencia media: 1.000.

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
| capital_social_actual | 1.000 | 1.000 | 1.000 | 1.000 |
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
| acta_a | ninguno | 0 | 0 | 0 | n/d | n/d | 3/3 | - | - |
| acta_b | capital_social_actual | 3 | 3 | 0 | 0.500 | 1.000 | 0/3 | socios (3) | - |
| acta_c | ninguno | 0 | 0 | 0 | n/d | n/d | 3/3 | - | - |
| total | - | 3 | 3 | 0 | 0.500 | 1.000 | 6/9 | socios (3) | - |

## Latencia por documento (ms)

| Acta | Media | p50 | Max |
|---|---|---|---|
| acta_a | 40980.2 | 36582.9 | 50032.0 |
| acta_b | 101309.7 | 101246.8 | 101977.4 |
| acta_c | 53357.0 | 54784.8 | 58591.5 |
| global | 65215.6 | 54784.8 | 101977.4 |

## Tokens por etapa (media por documento)

| Acta | Etapa | Llamadas | Tokens de entrada | Tokens de salida |
|---|---|---|---|---|
| acta_a | extraction | 1.00 | 5373.0 | 4856.0 |
| acta_a | vlm_reread | 0.00 | 0.0 | 0.0 |
| acta_a | decisions_predicate | 0.00 | 0.0 | 0.0 |
| acta_b | extraction | 2.00 | 9843.0 | 6970.0 |
| acta_b | vlm_reread | 3.00 | 6847.0 | 3111.3 |
| acta_b | decisions_predicate | 4.33 | 5680.3 | 0.0 |
| acta_c | extraction | 1.00 | 5780.0 | 6354.3 |
| acta_c | vlm_reread | 0.00 | 0.0 | 0.0 |
| acta_c | decisions_predicate | 0.00 | 0.0 | 0.0 |

## Costo

- Decisions API: 17041 tokens de entrada en total; a USD 0.10 por millon, costo USD 0.001704.
- Responses API (gpt-6-luna, extraccion y relectura VLM): 83529 tokens de entrada y 63875 de salida en total; no se asume precio.

## Errores

- Ninguno.
