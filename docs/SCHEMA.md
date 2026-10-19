# Esquema de salida

Este documento compara el esquema esperado por el ejercicio con el implementado en `src/schemas/deed_models.py` (`CorporateDeedOutput`, versión `1.0.0`) y justifica cada extensión o cambio de nombre. Los nombres de los ocho campos raíz se conservan exactamente como los pide el ejercicio.

## Correspondencia de campos

| Campo esperado | Tipo esperado | Implementado | Cambio |
|----------------|---------------|--------------|--------|
| `razon_social` | string o null | `FieldWithEvidence[str]` | Envoltura con evidencia |
| `fecha_constitucion` | date o null | `FieldWithEvidence[str]` con formato `YYYY-MM-DD` | Fecha como cadena ISO 8601 |
| `numero_escritura` | string o null | `FieldWithEvidence[str]` | Envoltura con evidencia |
| `notario` | `{nombre, numero}` o null | `FieldWithEvidence[NotaryInfo]`: `nombre`, `numero_notaria` (int), `entidad_federativa` | `numero` renombrado; entidad agregada |
| `socios` | array | `FieldWithEvidence[list[ShareholderItem]]`: `nombre`, `rfc`, `acciones`, `valor_aportacion`, `porcentaje`, `evidence` | Estructura por socio con evidencia propia |
| `representantes_legales` | array | `FieldWithEvidence[list[LegalRepresentativeItem]]`: `nombre`, `nombre_normalizado`, `cargo`, `vigente`, `fecha_nombramiento`, `evidence` | Nombre normalizado, vigencia y fecha |
| `poderes` | array | `FieldWithEvidence[list[PowerOfAttorneyItem]]`: `otorgado_a`, `tipo_poder`, `modalidad`, `vigente`, `evidence` | Un elemento por tipo de poder y persona |
| `capital_social_actual` | `{valor, moneda}` o null | `FieldWithEvidence[CapitalInfo]`: `monto_numerico`, `moneda`, `monto_en_letra`, `fecha_acto_vigente`, `escritura_origen` | `valor` renombrado; tres campos agregados |
| (no existe) | | `amendment_history: list[AmendmentEvent]` | Agregado |
| (no existe) | | `audit_metadata: AuditMetadata` | Agregado |

La compatibilidad con el contrato esperado es directa: `razon_social` equivale a `razon_social.value`, `notario.numero` a `notario.value.numero_notaria`, `capital_social_actual.valor` a `capital_social_actual.value.monto_numerico`, y un campo nulo del contrato esperado corresponde a `value: null` con `status` `no_encontrado` o `no_aplicable`.

## Justificación

**Envoltura `FieldWithEvidence`.** Cada campo lleva `value`, `status`, `evidence` y `requires_human_review` / `review_reason`. Un valor sin evidencia no es auditable: `evidence` guarda `page_number`, `source_quote` literal, `confidence_score`, `confidence_source` (`native_pdf`, `ocr_engine`, `decisions_predicate`, `deterministic_rule`; el valor `llm_logprobs` existe en el enum por compatibilidad con la especificación, pero el pipeline no lo produce), `grounding_verified` y `bounding_box` opcional. Esto permite al revisor ir directamente a la página y a la región, y permite a la evaluación medir la tasa de evidencia correcta (Evidence Hit Rate). Los elementos de las listas llevan su propia evidencia porque socios, representantes y poderes pueden estar en páginas distintas.

**Campos de evidencia agregados por la relectura y los puntajes.** Son opcionales (valor por defecto `null`), de modo que las cachés existentes siguen siendo válidas:

| Campo | Contenido | Justificación |
|-------|-----------|---------------|
| `ocr_reading` | Texto que leyó el OCR local en la región | Permite al revisor y a la auditoría comparar lo que el OCR leyó con lo que leyó el modelo de visión |
| `vlm_reading` | Lectura del modelo de visión sobre el recorte | Se conserva aunque no reemplace al texto OCR, para auditar una corrección |
| `vlm_legible` | `true` o `false` según el modelo de visión | `false` declara la región ilegible y envía el campo a revisión; el modelo no debe adivinar caracteres ocultos |
| `verification_probability` | Probabilidad `p` del predicado de Decisions | `confidence_score` guarda `abs(2p - 1)`; sin `p` no se puede distinguir un "sí" confiado de un "no" confiado |
| `crop_path` | Ruta del PNG del recorte (`outputs/crops/{acta}_p{page}_{field}_bbox.png`) | El revisor decide sobre el campo viendo exactamente la imagen que se verificó |
| `page_legibility_score` | `OCR_LEGIBILITY_SCORE` normalizado de la página (0 a 1) | Explica por qué un campo con confianza OCR alta se escaló: la confianza del OCR no detecta omisiones |

**`status` en lugar de solo null.** Un null no distingue entre "el dato no viene en el documento", "el dato es ilegible" y "el dato no aplica al régimen". `extraido`, `no_encontrado` (requerido pero ausente o ilegible) y `no_aplicable` (legalmente inaplicable) separan esos casos; solo `no_encontrado` en un campo requerido dispara revisión. El conjunto de campos requeridos y aplicables viene del `DocumentTypeSpec` registrado.

**Fechas como cadena ISO.** JSON no tiene tipo fecha. Se usa `YYYY-MM-DD` como cadena, que el esquema estricto de Structured Outputs acepta sin ambigüedad; los validadores y el resolvedor la convierten a `date` para comparar y para la verificación de coherencia temporal.

**`notario.numero_notaria` y `entidad_federativa`.** El número de notaría solo identifica al notario dentro de una entidad federativa: la Notaría 15 de Jalisco y la Notaría 15 de Ciudad de México son distintas. `numero_notaria` es entero para comparar sin ambigüedad de formato; el nombre explícito evita confundirlo con el número de escritura.

**`capital_social_actual`: `monto_numerico`, `monto_en_letra`, `fecha_acto_vigente`, `escritura_origen`.** `monto_numerico` es el `valor` esperado, con un nombre que lo distingue del monto en letra. `monto_en_letra` permite la validación determinística número frente a letra, que en una escritura es el control natural contra errores de lectura (por ejemplo, un sello sobre las cifras). `fecha_acto_vigente` y `escritura_origen` registran qué acto notarial fijó el capital vigente: en el Acta C el capital de 1,000,000.00 proviene de la escritura 58,400 del 2024-08-20, no de la constitución. Sin esos dos campos el valor "actual" no es verificable.

**Representantes: `nombre_normalizado`, `vigente`, `fecha_nombramiento`.** `nombre_normalizado` (mayúsculas, sin acentos ni títulos como `LIC.`, `SR.`, `ING.`) es la llave del cruce difuso entre nombramientos y revocaciones. `fecha_nombramiento` permite detectar una revocación fechada antes del nombramiento. `vigente` documenta el estado; en la salida final solo aparecen los vigentes.

**Poderes: un elemento por tipo de poder y persona.** Una revocación o una auditoría opera sobre el instrumento concreto (pleitos y cobranzas, administración, dominio). Agruparlos en un texto libre impediría revocar o comparar con granularidad. `modalidad` (individual o mancomunada) cambia quién puede firmar.

**`amendment_history`.** Un expediente puede contener la constitución y actos posteriores (aumentos de capital, otorgamientos y revocaciones de poderes). El LLM extrae los eventos con fecha y número de escritura, y el resolvedor determinístico calcula el estado actual. Separar los hechos (eventos) de la interpretación (estado vigente) permite auditar por qué una persona no aparece como representante. En las revocaciones se agregan `matched_canonical_entity` y `fuzzy_match_score`, que documentan con quién se cruzó el nombre revocado y con qué similitud.

**`revoked_appointment_evidence`.** Extensión opcional del evento de revocación (lista vacía por defecto, por lo que las cachés existentes siguen siendo válidas). Guarda la evidencia estructurada del nombramiento que la revocación eliminó, de modo que se conservan juntas la evidencia del nombramiento y la de la revocación, aunque la persona ya no esté en la lista activa.

**`audit_metadata`.** `job_id`, `doc_sha256`, `document_filename`, `schema_version`, `model_version`, `total_pages`, `native_pages_count`, `ocr_pages_count`, `global_human_review_required`, `processing_time_ms` y `security_flags`. Vincula el resultado con el binario exacto, con la versión del esquema y del modelo que lo produjo, y con la ruta de procesamiento. `global_human_review_required` permite a un consumidor decidir sin recorrer todos los campos. `security_flags` es una lista (vacía por defecto) de señales de seguridad a nivel de documento, por ejemplo `prompt_injection_suspected`, `prompt_injection_check_refused` o `prompt_injection_check_unavailable`: permite a un consumidor saber que una página pudo contener instrucciones dirigidas a un sistema automático sin recorrer los campos; la bandera de inyección sospechada fuerza `global_human_review_required`. Las entradas de caché generadas en vivo guardan además `prompt_sha256`.

## Versionado

`schema_version` sigue versionado semántico. Agregar un campo opcional con valor por defecto (como `revoked_appointment_evidence`, los campos de relectura y puntaje de `Evidence` o `security_flags`) es compatible hacia atrás; renombrar o eliminar un campo, o cambiar su tipo, requiere una versión mayor y un periodo en que los consumidores lean ambas. Los consumidores deben validar `schema_version` antes de interpretar el JSON.
