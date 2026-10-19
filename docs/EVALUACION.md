# Evaluación

Este documento describe cómo se evalúa la calidad de la extracción, qué resultados se obtuvieron con los tres documentos de prueba, por qué los errores no pesan igual, cuándo el sistema se abstiene y envía a revisión humana, y qué haría falta para una evaluación representativa de producción. La implementación está en `src/evaluation/evaluator.py`; el reporte se genera con `actas-cli evaluate` en `outputs/evaluation_report.json` (copia en [`results/evaluation_report.json`](../results/evaluation_report.json)). El ground truth legible está en [GROUND_TRUTH.md](GROUND_TRUTH.md).

## 1. Métricas por tipo de campo

No se usa una exactitud global: cada tipo de campo tiene una métrica acorde a su naturaleza y un peso acorde a su impacto en el negocio. Los pesos suman 1.0.

| Tipo | Campo | Métrica | Aprobado si | Peso |
|------|-------|---------|-------------|------|
| Fecha | `fecha_constitucion` | Coincidencia exacta ISO `YYYY-MM-DD` | igual | 0.10 |
| Identificador | `numero_escritura` | Coincidencia exacta tras normalizar a dígitos (`45,120` = `45120`) | igual | 0.10 |
| Monto | `capital_social_actual` | Coincidencia numérica con tolerancia 0.01 | diferencia <= 0.01 | 0.20 |
| Texto libre | `razon_social` | `token_sort_ratio / 100` de RapidFuzz sobre texto normalizado | >= 0.90 | 0.10 |
| Entidad compuesta | `notario` | Número de notaría exacto combinado con similitud del nombre | número igual y nombre similar | 0.10 |
| Lista de entidades | `socios` | F1 de conjunto sobre nombres (cruce >= 85) más exactitud de porcentajes | F1 y porcentajes correctos | 0.10 |
| Lista crítica | `representantes_legales` | Precisión, recall y F1 de conjunto sobre representantes vigentes | F1 = 1 | 0.20 |
| Lista de instrumentos | `poderes` | F1 de conjunto sobre pares (apoderado, tipo de poder) vigentes | F1 = 1 | 0.10 |

Medidas adicionales:

- **Evidence Hit Rate.** Proporción de campos extraídos cuya evidencia cita la página correcta del ground truth y tiene `grounding_verified = true`. Mide trazabilidad, no solo el valor.
- **Coincidencia de enrutamiento a revisión.** El conjunto de campos marcados con `requires_human_review` debe ser igual al esperado (Acta B: `capital_social_actual`; Acta A y C: ninguno). Marcar de más cuesta trabajo manual; marcar de menos deja pasar un dato no verificado.
- **Falso positivo crítico.** Un representante revocado que aparece como vigente resta 0.25 a la calificación del documento (con piso en 0) y emite `[CRITICAL_RISK]`; además, si existe al menos uno en el lote, se resta 0.25 una vez a la calificación global (`global_penalty`).

## 2. Resultados

### Ejecución determinística (caché preverificada)

Resultado de `actas-cli demo` sin credenciales (archivos en [`results/`](../results/)):

| Documento | Calificación ponderada | Penalización | Final | Evidence Hit Rate | Falsos positivos críticos | Revisión: esperada / marcada |
|-----------|------------------------|--------------|-------|-------------------|---------------------------|------------------------------|
| Acta A | 1.000 | 0.00 | 1.000 | 100.0% | 0 | ninguna / ninguna |
| Acta B | 1.000 | 0.00 | 1.000 | 100.0% | 0 | `capital_social_actual` / `capital_social_actual` |
| Acta C | 1.000 | 0.00 | 1.000 | 100.0% | 0 | ninguna / ninguna |
| Global | 1.000 | 0.00 | 1.000 | 100.0% | 0 | coincide en los tres |

Los ocho campos de los tres documentos aprueban su métrica (24 de 24). En el Acta B el motivo registrado es: `Low OCR confidence (0.72) and Decisions predicate confidence (0.56 < 0.80) due to overlapping notary seal on page 2 bbox [72.0, 410.0, 520.0, 455.0].` En el Acta C `SR. JUAN C. PEREZ LOPEZ` se cruza con `LIC. JUAN CARLOS PÉREZ LÓPEZ` con similitud 89.41 y se elimina de la lista vigente.

Esta ejecución valida la lógica determinística (resolución temporal, validadores, enrutamiento y módulo de evaluación) sobre una extracción conocida. No mide la calidad de los modelos: la extracción y la verificación vienen de la caché preverificada, cuyo valor `p = 0.78` es la simulación del escenario especificado.

### Ejecución en vivo (gpt-6-luna)

Mediciones del 2026-10-18 con `gpt-6-luna`. Extracción (Responses API) y verificación (Decisions API) en vivo; el texto OCR se sirvió desde la caché OCR determinística porque RapidOCR no estaba instalado.

| Prueba | Resultado |
|--------|-----------|
| Extracción Acta A | Salida válida contra el esquema; 0 fallas de grounding en 20 evidencias; calificación final 1.000; 5,087 tokens de entrada y 5,101 de salida |
| Extracción Acta B | Salida válida contra el esquema; 0 fallas de grounding en 15 evidencias; calificación final 1.000; 4,656 tokens de entrada y 3,323 de salida |
| Ejecuciones anteriores, antes de ajustar el prompt | Calificaciones 0.9667, 0.9 y 0.9849. Las diferencias venían del nombre canónico del tipo de poder y de la abreviatura del régimen en `razon_social`; se corrigieron con reglas explícitas en el prompt |
| Decisions, recorte del capital del Acta B con texto OCR | `p = 0.35`, confianza calibrada 0.30 < 0.80, 867 tokens de entrada; `capital_social_actual` enviado a revisión |
| Decisions, mismo recorte sin texto OCR | `p = 0.40`, calibrada 0.20, 583 tokens |
| Decisions, solo texto OCR sin imagen | `p = 0.99`, calibrada 0.98, 579 tokens |
| Decisions, versión anterior con sello semitransparente (monto legible) | `p = 0.72` con texto OCR; `p = 0.90` solo imagen |
| Latencia observada de Decisions (9 llamadas) | 159 a 680 ms |

Lectura de los resultados: el verificador real duda (y se inclina por "no") cuando la tinta efectivamente oculta las cifras, y es confiado cuando el monto es legible. Con solo el texto OCR responde con `p = 0.99`, lo que confirma que el texto OCR no debe tratarse como prueba y que el recorte de imagen es indispensable en la solicitud. La extracción se muestrea con la temperatura por defecto del modelo, por lo que la calificación puede variar entre ejecuciones; con tres documentos estas cifras no son una estimación estadística de la tasa de error.

### Verificación Decisions: `test-decisions` (2026-10-19)

`actas-cli test-decisions --live` ejecuta cuatro casos contra la API Decisions real; con `--no-live` las respuestas son simuladas y el reporte las declara (`simulated: true`). Reportes: [`results/decisions_verification_report_live.json`](../results/decisions_verification_report_live.json) y [`results/decisions_verification_report_simulated.json`](../results/decisions_verification_report_simulated.json).

| Caso | En vivo | Simulado | Veredicto esperado |
|------|---------|----------|--------------------|
| Predicado sobre texto limpio del capital | `p = 1.00` | `p = 0.98` | CONFIRMED |
| Predicado sobre el recorte sellado del Acta B | `p = 0.48`, `abs(2p - 1) = 0.04` | `p = 0.78`, 0.56 | REVIEW |
| Predicado de contradicción: representante revocado del Acta C | `p = 0.00` | `p = 0.03` | REJECTED |
| Riesgo de inyección de instrucciones | normalizado 0.997 (coincidencias de regex `ignore_previous_instructions`, `force_approval`, `skip_human_review`) | normalizado 0.9367 | FLAGGED |

Los cuatro casos en vivo usaron 1,891 tokens de entrada (unos USD 0.0002). Una solicitud de legibilidad e inyección de página completa usa unos 4.5 mil tokens de entrada (unos USD 0.00045).

### Relectura VLM y Acta B completa en vivo con RapidOCR real

Sesión en vivo, octubre de 2026, `gpt-6-luna`:

- **Relectura VLM del recorte sellado:** `{"reading": "QUINTA.- El capital social es de $ MIL PESOS 00/100 M.N.).", "legible": false}`. Sobre un recorte limpio del Acta A la lectura fue exacta con `legible = true`.
- **Hallazgo sobre el OCR real:** RapidOCR lee la página sellada del Acta B con confianza mayor o igual a 0.90 y omite en silencio `$250,000.00 (DOSCIENTOS CINCUENTA`; la confianza del Nivel 1 no detecta omisiones y el umbral de 0.85 nunca se dispara. La legibilidad normalizada de Decisions sí lo detecta: 0.33 en la página 2 del Acta B frente a 0.997 en la página 1 del Acta A, con umbral 0.75.
- **Flujo completo:** legibilidad 0.33; la primera extracción dejó capital y `poderes` como `no_encontrado`; la relectura VLM de página completa devolvió `legible = false` con la transcripción visible; una reextracción obtuvo capital 250,000.00 con `monto_en_letra` `[ilegible] MIL PESOS 00/100 M.N.`; Decisions sobre el recorte grueso de página completa dio `p = 0.98` (frente a 0.48 con el recorte ajustado); el capital se envió a revisión por la discrepancia número frente a letra y `legible = false`.
- **Limitación:** la fusión a página completa entrega al predicado un recorte grueso, por lo que la revisión provino de las verificaciones determinísticas y no del predicado. RapidOCR se ejercitó en vivo una sola vez; no hay una evaluación más amplia de OCR ni de relectura VLM.

## 3. Asimetría de errores

Los errores no se ponderan igual porque sus consecuencias no son equivalentes:

1. **Representante revocado reportado como vigente (falso positivo crítico).** Permite que una persona sin facultades firme, abra cuentas o celebre contratos a nombre de la sociedad. El daño es legal y económico, y no se detecta después porque el dato "parece" correcto. Por eso tiene una penalización explícita que domina la calificación en lugar de diluirse entre campos correctos.
2. **Representante vigente omitido (falso negativo).** Genera fricción: el expediente queda incompleto y una persona lo corrige. Es recuperable y visible. Se refleja en el recall de `representantes_legales`, sin penalización adicional.
3. **Capital social o número de escritura erróneo.** Un monto equivocado afecta reglas de riesgo y la coherencia del expediente; por eso el capital pesa 0.20 y se valida contra el monto en letra. Un error en `razon_social` por puntuación es menor y se tolera con similitud >= 0.90.
4. **Valor correcto sin evidencia verificable.** No es un error de valor, pero sí de auditoría: si la cita no está en la página, el dato no se puede defender. Se mide por separado con Evidence Hit Rate y la confianza se topa en 0.40.

En términos de umbrales, la consecuencia es que para campos críticos se prefiere enviar a revisión (costo: minutos de un analista) antes que aceptar un valor dudoso (costo: un acto inválido).

## 4. Criterios de abstención y revisión humana implementados

El sistema no "adivina": cuando la evidencia no alcanza, conserva el valor extraído, marca `requires_human_review = true` y escribe un `review_reason` explícito. Criterios activos en el prototipo (`src/core/config.py`, `src/pipeline/validators.py`, `src/pipeline/resolver.py`, `src/pipeline/orchestrator.py`):

| Criterio | Regla | Efecto |
|----------|-------|--------|
| Confianza OCR | Región de evidencia con confianza < 0.85 (en citas multilínea, el mínimo de las regiones que la cubren) | Recorte, relectura VLM y Tier 2 |
| Legibilidad de la página | `OCR_LEGIBILITY_SCORE` normalizado < 0.75 (`PAGE_LEGIBILITY_THRESHOLD`) en una página escaneada | Los campos críticos de la página escalan a relectura VLM y predicado, aunque la confianza OCR sea alta |
| Relectura VLM | La relectura declara `legible = false`, o falla | Revisión |
| Relectura tardía acotada | Campo crítico requerido `no_encontrado`, o falla de valor en cita o de coherencia del capital en página escaneada | Una relectura y a lo sumo una reextracción; si no se resuelve, revisión con motivo factual |
| Inyección de instrucciones | Prefiltro de regex (`src/pipeline/injection_guard.py`) o `PROMPT_INJECTION_RISK_SCORE` normalizado >= 0.5 (`PROMPT_INJECTION_RISK_THRESHOLD`) | `security_flags = ["prompt_injection_suspected"]`, revisión global forzada, sin aprobación automática de los campos de esa página |
| Verificación por predicado | Confianza calibrada `abs(2p - 1)` < 0.80 | Revisión |
| Verificador no disponible | Rechazo, respuesta ausente o error de la API Decisions | Revisión con motivo explícito |
| Grounding | La cita no aparece en la página citada (coincidencia normalizada exacta, o `partial_ratio` >= 90 con guardas de dígitos y de palabras) | Confianza topada en 0.40 y revisión; el tope no se levanta con escalamientos posteriores |
| Valor en cita | En `numero_escritura`, `fecha_constitucion`, `notario` y `capital_social_actual`, el valor no aparece en su propia cita | Revisión |
| Monto número frente a letra | `monto_numerico` no corresponde a `monto_en_letra` | Revisión |
| Coherencia temporal | Evento de modificación fechado antes de `fecha_constitucion`; revocación anterior al nombramiento | Revisión; el representante no se elimina en silencio |
| Cruce de revocaciones | Similitud >= 85 con apellidos paterno y materno idénticos y primer nombre coincidente revoca; banda 75 a 85, o apellidos distintos | Revoca, o revisión sin decidir en silencio |
| Campo requerido ausente | Requerido por el `DocumentTypeSpec` y sin valor | `no_encontrado` y revisión |
| Confianza genérica | Cualquier evidencia con confianza < 0.80 | Revisión |

Además, cuando las lecturas OCR y VLM discrepan, la corrección solo se acepta si Decisions la confirma con `abs(2p - 1)` >= 0.80.

Los umbrales son valores iniciales razonados, no calibrados. Se calibraron solo contra datos sintéticos.

## 5. Datos necesarios para una evaluación representativa de producción

- **Muestra real y estratificada.** Documentos reales anonimizados, estratificados por notaría y entidad federativa, década de otorgamiento, régimen (S.A. de C.V., S. de R.L. de C.V., S.A.P.I., S.C.), nativo frente a escaneado, calidad de escaneo y fotografías de teléfono.
- **Casos difíciles con sobremuestreo.** Expedientes con varias modificaciones, revocaciones múltiples, homónimos y casi homónimos, sellos y firmas sobre cifras, páginas rotadas o faltantes, anexos manuscritos y texto que intente instruir al modelo.
- **Etiquetado con evidencia.** Cada valor con su página y región, por al menos dos etiquetadores con adjudicación; medir acuerdo entre etiquetadores para conocer el techo alcanzable.
- **Resultado de la revisión humana.** Registrar qué corrigió el analista en cada caso enviado a revisión: es la fuente para calibrar umbrales y medir la tasa de falsos negativos del enrutamiento.
- **Tamaño suficiente para intervalos de confianza.** Con tres documentos no hay estimación estadística; para afirmar una tasa de falsos positivos críticos baja se necesitan cientos de expedientes con revocaciones.
- **Separación de conjuntos.** Deduplicación por SHA-256 y por casi duplicados; partición por cliente o expediente y temporal, para que el conjunto de prueba no contenga documentos de las mismas sociedades usadas para ajustar prompts o umbrales.

## 6. Si un LLM devuelve `confidence = 0.97`

Un número de confianza escrito por el modelo dentro de su respuesta es texto generado, no una probabilidad. No hay garantía de que, entre las respuestas con 0.97, el 97% sean correctas; los modelos tienden a ser sobreconfiados y el valor cambia con el prompt. Por eso el pipeline no usa confianzas verbalizadas para decidir.

Lo que sí usa el prototipo es la probabilidad `p` que devuelve la API Decisions para una pregunta binaria concreta ("¿el recorte muestra $250,000.00?"), mapeada a `abs(2p - 1)`; ese endpoint es la primitiva diseñada para devolver una probabilidad por pregunta (o un `score` con `confidence`) sin pedirle al modelo que escriba un número (ver [ARCHITECTURE_AND_DESIGN.md](ARCHITECTURE_AND_DESIGN.md), sección 4). También usa fuentes determinísticas (texto nativo 1.0, confianza del motor OCR, reglas). Aun así, `p` tampoco se considera calibrada para este dominio hasta medirla.

Cómo determinar si una confianza es útil:

1. **Datos etiquetados.** Reunir pares (confianza, correcto o incorrecto) por campo y por tipo documental, sobre documentos que no se usaron para ajustar el sistema.
2. **Calibración.** Diagrama de confiabilidad (exactitud observada por intervalo de confianza), error de calibración esperado (ECE) y puntuación de Brier. Si la curva se aleja de la diagonal, recalibrar con regresión isotónica o Platt sobre un conjunto separado.
3. **Discriminación.** Que la confianza ordene bien aciertos y errores (AUROC, curva de cobertura frente a exactitud): una confianza calibrada pero sin poder de separación no sirve para enrutar.
4. **Umbral por costo.** Elegir el umbral que minimiza el costo esperado según el costo de un falso positivo (aceptar un valor erróneo) y de un falso negativo (enviar a revisión un valor correcto), por campo: para representantes y capital se acepta más revisión que para `razon_social`.
5. **Monitoreo.** Repetir la medición por versión de modelo y de prompt, y vigilar deriva (tasa de revisión, distribución de confianzas, correcciones de analistas). Un cambio de modelo invalida la calibración anterior.

## 7. Criterios para considerar el sistema listo para producción

Criterios a acordar con las áreas de negocio y riesgo, medidos sobre un conjunto real de prueba separado:

- Cero falsos positivos críticos en el conjunto de prueba, con un intervalo de confianza cuyo límite superior sea aceptable para riesgo.
- Exactitud por campo, no global, por encima del objetivo de cada campo, medida solo sobre los campos que el sistema aceptó sin revisión.
- Tasa de revisión humana compatible con la capacidad operativa, y tasa de errores entre los campos no enviados a revisión por debajo del límite acordado.
- Evidence Hit Rate cercano a 100% en los campos aceptados.
- Umbrales calibrados con datos etiquetados reales y documentados por versión de modelo y prompt.
- Pruebas de inyección de instrucciones y de documentos adversarios sin cambios en el resultado.
