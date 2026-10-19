# Ground truth de los documentos de prueba

Versión legible de `data/ground_truth/ground_truth.json`. Los tres documentos (`data/samples/acta_a.pdf`, `acta_b.pdf`, `acta_c.pdf`) son sintéticos y se generan de forma determinística con `actas-cli generate-samples`; personas, sociedades, RFC y notarías son ficticios. Los RFC se muestran enmascarados con la misma regla que aplica la bitácora de auditoría. La columna "Página" indica la página donde debe encontrarse la evidencia.

## Resumen de escenarios

| Documento | Páginas | Ruta esperada | Revisión humana esperada | Representantes revocados |
|-----------|---------|---------------|--------------------------|--------------------------|
| Acta A | 2 | Páginas 1 y 2 nativas | Ninguna | Ninguno |
| Acta B | 2 | Página 1 nativa, página 2 escaneada | `capital_social_actual` | Ninguno |
| Acta C | 4 | Páginas 1 a 4 nativas | Ninguna | `JUAN CARLOS PEREZ LOPEZ` |

## Acta A: constitución nativa limpia

SHA-256: `822992832a9a659892b2e47ef53f0b8a01e64af1597355bfe86b8aa28f69c697`

| Campo | Valor esperado | Página |
|-------|----------------|--------|
| `razon_social` | TECNOLOGIA FINANCIERA DEL VALLE, S.A. DE C.V. | 1 |
| `fecha_constitucion` | 2021-03-15 | 1 |
| `numero_escritura` | 45,120 | 1 |
| `notario` | ROBERTO MENDEZ LARA, notaría 88, Ciudad de Mexico | 1 |
| `capital_social_actual` | 500,000.00 MXN (QUINIENTOS MIL PESOS 00/100 MONEDA NACIONAL), vigente desde 2021-03-15, escritura 45,120 | 1 |

| Socio | RFC | Acciones | Porcentaje | Página |
|-------|-----|----------|------------|--------|
| CARLOS EDUARDO RAMIREZ SOTO | RASC******HM1 | 300 | 60.0 | 2 |
| SOFIA VALENTINA HERRERA CRUZ | HECS******MJ2 | 200 | 40.0 | 2 |

| Representante legal | Cargo | Nombramiento | Página |
|---------------------|-------|--------------|--------|
| CARLOS EDUARDO RAMIREZ SOTO | Administrador Unico | 2021-03-15 | 2 |
| SOFIA VALENTINA HERRERA CRUZ | Apoderada General | 2021-03-15 | 2 |

| Apoderado | Poderes (modalidad individual) | Página |
|-----------|--------------------------------|--------|
| CARLOS EDUARDO RAMIREZ SOTO | Pleitos y Cobranzas; Actos de Administración; Actos de Dominio | 2 |
| SOFIA VALENTINA HERRERA CRUZ | Pleitos y Cobranzas; Actos de Administración | 2 |

Historial: 1 evento (`constitucion_original`, 2021-03-15, escritura 45,120, capital 500,000.00).

## Acta B: documento híbrido con sello sobre el capital

SHA-256: `06ccca91d22921abd4d8578b2eb26a7520641fc5c4310bf5c719cf9d8f10dacf`

Escenario: la página 2 es una imagen con un sello notarial cuya tinta cubre el monto `$250,000.00` y parte del monto en letra. Se espera confianza OCR 0.72 (menor a 0.85), escalamiento a verificación por predicado y envío de `capital_social_actual` a revisión humana.

| Campo | Valor esperado | Página |
|-------|----------------|--------|
| `razon_social` | COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. DE C.V. | 1 |
| `fecha_constitucion` | 2020-07-22 | 1 |
| `numero_escritura` | 12,890 | 1 |
| `notario` | FERNANDO GARZA TREVINO, notaría 42, Nuevo Leon | 1 |
| `capital_social_actual` | 250,000.00 MXN (DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.), vigente desde 2020-07-22, escritura 12,890 | 2 |

| Socio | RFC | Partes sociales | Porcentaje | Página |
|-------|-----|-----------------|------------|--------|
| JORGE ALBERTO SALINAS MORA | no consta | 150 | 60.0 | 2 |
| PATRICIA EUGENIA LOZANO VELA | no consta | 100 | 40.0 | 2 |

| Representante legal | Cargo | Nombramiento | Página |
|---------------------|-------|--------------|--------|
| JORGE ALBERTO SALINAS MORA | Gerente General | 2020-07-22 | 2 |

| Apoderado | Poderes (modalidad individual) | Página |
|-----------|--------------------------------|--------|
| JORGE ALBERTO SALINAS MORA | Pleitos y Cobranzas; Actos de Administración | 2 |

Historial: 1 evento (`constitucion_original`, 2020-07-22, escritura 12,890, capital 250,000.00).

Revisión humana esperada: `capital_social_actual`. El valor es correcto, pero no es verificable con confianza suficiente en la imagen; el resultado correcto es extraerlo y marcarlo para revisión, no omitirlo ni aceptarlo sin control.

## Acta C: constitución 2019 y asamblea protocolizada 2024

SHA-256: `f33303b2365630a8d0704d285ccb697988c51807459d010e3fbe7cb3c0cb31fa`

Escenario: páginas 1 y 2 contienen la constitución de 2019; páginas 3 y 4 la protocolización de una asamblea extraordinaria de 2024 con aumento de capital, revocación del administrador original y nombramiento de uno nuevo.

| Campo | Valor esperado | Página |
|-------|----------------|--------|
| `razon_social` | LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V. | 1 |
| `fecha_constitucion` | 2019-05-10 | 1 |
| `numero_escritura` | 30,100 (escritura constitutiva) | 1 |
| `notario` | ARTURO VILLASEÑOR ORTIZ, notaría 15, Jalisco | 1 |
| `capital_social_actual` | 1,000,000.00 MXN (UN MILLON DE PESOS 00/100 MONEDA NACIONAL), vigente desde 2024-08-20, escritura 58,400 | 3 |

| Socio | RFC | Acciones | Porcentaje | Página |
|-------|-----|----------|------------|--------|
| GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V. | no consta | 800 | 80.0 | 1 |
| MARIA ELENA TORRES VEGA | no consta | 200 | 20.0 | 1 |

| Representante legal vigente | Cargo | Nombramiento | Página |
|-----------------------------|-------|--------------|--------|
| MARIA ELENA TORRES VEGA | Apoderada Legal | 2019-05-10 | 2 |
| ING. ALEJANDRO GÓMEZ RUIZ | Administrador Unico | 2024-08-20 | 4 |

| Apoderado vigente | Poderes (modalidad individual) | Página |
|-------------------|--------------------------------|--------|
| MARIA ELENA TORRES VEGA | Pleitos y Cobranzas; Actos de Administración | 2 |
| ALEJANDRO GOMEZ RUIZ | Pleitos y Cobranzas; Actos de Administración; Actos de Dominio | 4 |

Historial esperado (4 eventos, orden cronológico):

| Evento | Fecha | Escritura | Sujeto afectado | Monto nuevo | Página |
|--------|-------|-----------|-----------------|-------------|--------|
| `constitucion_original` | 2019-05-10 | 30,100 | | 100,000.00 | 1 |
| `aumento_capital` | 2024-08-20 | 58,400 | | 1,000,000.00 | 3 |
| `revocacion_poderes` | 2024-08-20 | 58,400 | SR. JUAN C. PEREZ LOPEZ | | 3 |
| `otorgamiento_poderes` | 2024-08-20 | 58,400 | ING. ALEJANDRO GÓMEZ RUIZ | | 4 |

Representante revocado: `JUAN CARLOS PEREZ LOPEZ` (nombrado en 2019 como `LIC. JUAN CARLOS PÉREZ LÓPEZ`, revocado en 2024 como `SR. JUAN C. PEREZ LOPEZ`). No debe aparecer en `representantes_legales` ni en `poderes`; si aparece, el módulo de evaluación lo cuenta como falso positivo crítico. La evidencia de su nombramiento y de su revocación se conserva en `amendment_history`.

Puntos que este documento prueba: el capital vigente es el del acto más reciente (no el de la constitución); el nombre revocado está escrito con otro título, con inicial y sin acentos, por lo que el cruce requiere normalización y similitud difusa; el número de escritura del campo raíz sigue siendo el de la constitución.
