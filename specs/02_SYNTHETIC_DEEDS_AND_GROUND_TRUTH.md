# Specification 02: Synthetic Deeds (Acta A, B, C), Ground Truth & Deterministic Cache

## CRITICAL SYNCHRONIZATION RULE
When `src/generators/synthetic_deeds.py` generates `acta_a.pdf`, `acta_b.pdf`, and `acta_c.pdf`:
1. Use `reportlab` with `invariant=1` so PDF binaries are deterministic.
2. Immediately after writing each PDF, compute its actual `SHA-256` hash (`hashlib.sha256(pdf_bytes).hexdigest()`).
3. Automatically write the pre-verified extraction payload into `data/cache/{sha256}.json` AND update `data/ground_truth/ground_truth.json`.
This guarantees 100% synchronization between the PDF text, `source_quote` substrings, `SHA-256` cache keys, and `ground_truth.json`.

---

## 1. Exact Specification for `acta_a.pdf` (Clean Digital Native PDF - 2 Pages)
- **Scenario:** Clean digital native incorporation deed. All pages have >= 50 clean characters in `PyMuPDF`. Cost of OCR = $0.
- **Page 1 Text (Must contain these exact verbatim clauses):**
  - Header: `INSTRUMENTO PUBLICO NUMERO 45,120 (CUARENTA Y CINCO MIL CIENTO VEINTE).`
  - Date & Place: `En la Ciudad de Mexico, a los 15 dias del mes de marzo de 2021, yo, el Licenciado ROBERTO MENDEZ LARA, Titular de la Notaria Publica Numero 88 de la Ciudad de Mexico, hago constar el CONTRATO DE SOCIEDAD MERCANTIL.`
  - Corporate Name: `PRIMERA.- DENOMINACION: La sociedad se denominara "TECNOLOGIA FINANCIERA DEL VALLE", seguida de las palabras SOCIEDAD ANONIMA DE CAPITAL VARIABLE, o de su abreviatura "S.A. DE C.V."`
  - Capital Clause: `SEXTA.- CAPITAL SOCIAL: El capital social fijo es la cantidad de $500,000.00 (QUINIENTOS MIL PESOS 00/100 MONEDA NACIONAL), representado por 500 acciones ordinarias nominativas con valor nominal de $1,000.00 cada una.`
- **Page 2 Text (Must contain these exact verbatim clauses):**
  - Shareholders (`TRANSITORIA PRIMERA`):
    - `1. CARLOS EDUARDO RAMIREZ SOTO, con RFC RASC850412HM1, suscribe 300 acciones con valor de $300,000.00 (60.0%).`
    - `2. SOFIA VALENTINA HERRERA CRUZ, con RFC HECS901105MJ2, suscribe 200 acciones con valor de $200,000.00 (40.0%).`
  - Legal Representatives & Powers (`TRANSITORIA SEGUNDA`):
    - `Se designa como ADMINISTRADOR UNICO al señor CARLOS EDUARDO RAMIREZ SOTO, a quien se le confiere PODER GENERAL PARA PLEITOS Y COBRANZAS, PODER GENERAL PARA ACTOS DE ADMINISTRACION Y PODER GENERAL PARA ACTOS DE DOMINIO, para ser ejercido de manera INDIVIDUAL.`
    - `Se designa como APODERADA GENERAL a la señora SOFIA VALENTINA HERRERA CRUZ, a quien se le confiere PODER GENERAL PARA PLEITOS Y COBRANZAS Y ACTOS DE ADMINISTRACION, para ser ejercido de manera INDIVIDUAL.`

---

## 2. Exact Specification for `acta_b.pdf` (Noisy Scanned / Illegible Seal - 2 Pages)
- **Scenario:** Hybrid/Scanned deed with OCR uncertainty on Page 2 caused by an overlapping notary ink seal over the Capital and Notary Number, triggering Escalation Tier 1 (OCR confidence `0.72` < `0.85`) and Escalation Tier 2 (`Decisions` Predicate verifier returning `p = 0.78`, mapped confidence `|2(0.78) - 1| = 0.56` < `0.80`), which sets `requires_human_review = True`.
- **Page 1 (Digital Native Text):**
  - `ESCRITURA PUBLICA NUMERO 12,890 (DOCE MIL OCHOCIENTOS NOVENTA).`
  - `En la Ciudad de Monterrey, Estado de Nuevo Leon, a 22 de julio de 2020, ante mi, Licenciado FERNANDO GARZA TREVINO, Titular de la Notaria Publica Numero 42 del Primer Distrito Registral en el Estado de Nuevo Leon.`
  - `CLAUSULA PRIMERA.- La sociedad mercantil adoptara la denominacion de "COMERCIALIZADORA INDUSTRIAL DEL NORTE, S. DE R.L. DE C.V."`
- **Page 2 (Rendered as a Raster Image via Pillow embedded in PDF with simulated stamp noise, 0 native text chars in PyMuPDF):**
  - Visual content rendered onto image:
    - Capital clause obscured by simulated diagonal stamp `[SELLO NOTARIAL ILEGIBLE - COPIA CERTIFICADA]`: `QUINTA.- El capital social es de $250,000.00 (DOSCIENTOS CINCUENTA MIL PESOS 00/100 M.N.).`
    - Shareholders: `JORGE ALBERTO SALINAS MORA (150 partes sociales, $150,000.00, 60.0%) y PATRICIA EUGENIA LOZANO VELA (100 partes sociales, $100,000.00, 40.0%).`
    - Legal Representative: `GERENTE GENERAL: JORGE ALBERTO SALINAS MORA con PODER GENERAL PARA PLEITOS Y COBRANZAS Y ACTOS DE ADMINISTRACION (Individual).`
  - **Expected Pipeline Behavior for Acta B:**
    - `PyMuPDF` detects Page 1 as `NATIVE_PDF` (>= 50 chars) and Page 2 as `SCANNED_PAGE` (0 native chars).
    - Page 2 routes to `LocalOCRProvider` -> detects low confidence (`0.72`) on `capital_social_actual` bounding box `[72.0, 410.0, 520.0, 455.0]`.
    - Triggers `DecisionsVerifierProvider.verify_predicate()` -> returns `probability = 0.78`, calibrated confidence `|2*0.78 - 1| = 0.56`.
    - Sets `capital_social_actual.requires_human_review = True` and `review_reason = "Low OCR confidence (0.72) and Decisions predicate confidence (0.56 < 0.80) due to overlapping notary seal on page 2 bbox [72.0, 410.0, 520.0, 455.0]."`.

---

## 3. Exact Specification for `acta_c.pdf` (Original Incorporation + Subsequent Amendment - 4 Pages)
- **Scenario:** Multi-act corporate file containing the original 2019 incorporation deed (Pages 1-2) and a 2024 Extraordinary General Assembly protocolization (Pages 3-4) that increases capital, revokes an original representative using an abbreviated name without accents, and appoints a new representative.
- **Page 1 (Original Incorporation - 2019-05-10):**
  - `ESCRITURA PUBLICA NUMERO 30,100 (TREINTA MIL CIEN).`
  - `En la Ciudad de Guadalajara, Jalisco, a 10 de mayo de 2019, ante el Licenciado ARTURO VILLASEÑOR ORTIZ, Notario Publico Numero 15 de Guadalajara, Jalisco, se constituye "LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V."`
  - `CLAUSULA QUINTA (CAPITAL ORIGINAL): El capital social inicial se fija en la suma de $100,000.00 (CIEN MIL PESOS 00/100 MONEDA NACIONAL).`
  - Shareholders: `GRUPO INVERSIONISTA DEL OCCIDENTE, S.A. DE C.V. (800 acciones, $80,000.00, 80.0%) y MARIANAELENA TORRES VEGA / MARIA ELENA TORRES VEGA (200 acciones, $20,000.00, 20.0%).` (Use exact string: `MARIA ELENA TORRES VEGA`).
- **Page 2 (Original Appointments - 2019-05-10):**
  - `TRANSITORIA SEGUNDA (NOMBRAMIENTOS ORIGINALES 2019):`
  - `1. Se nombra como ADMINISTRADOR UNICO al LIC. JUAN CARLOS PÉREZ LÓPEZ, otorgandole PODER GENERAL PARA PLEITOS Y COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO de forma INDIVIDUAL.`
  - `2. Se nombra como APODERADA LEGAL a MARIA ELENA TORRES VEGA, otorgandole PODER GENERAL PARA PLEITOS Y COBRANZAS Y ACTOS DE ADMINISTRACION de forma INDIVIDUAL.`
- **Page 3 (Extraordinary Assembly Amendment - 2024-08-20 - Capital Increase & Revocation):**
  - `ESCRITURA PUBLICA NUMERO 58,400 (CINCUENTA Y OCHO MIL CUATROCIENTOS).`
  - `En la Ciudad de Guadalajara, Jalisco, a 20 de agosto de 2024, ante el Licenciado ARTURO VILLASEÑOR ORTIZ, Notario Publico Numero 15 de Guadalajara, comparece el Delegado Especial para protocolizar el Acta de Asamblea General Extraordinaria de Accionistas de "LOGISTICA TRANSCONTINENTAL DEL PACIFICO, S.A. DE C.V."`
  - `PRIMER ACUERDO (AUMENTO DE CAPITAL SOCIAL): Se aprueba aumentar el capital social fijo de la suma de $100,000.00 M.N. para quedar fijado en la cantidad actual de $1,000,000.00 (UN MILLON DE PESOS 00/100 MONEDA NACIONAL).`
  - `SEGUNDO ACUERDO (REVOCACION DE PODERES Y CARGO): Se revoca en este acto el nombramiento de Administrador Unico, asi como todos y cada uno de los poderes generales para pleitos y cobranzas, actos de administracion y actos de dominio otorgados previamente en la escritura 30,100 a favor del SR. JUAN C. PEREZ LOPEZ, quedando sin efecto legal alguno su representacion.`
- **Page 4 (New Appointment - 2024-08-20):**
  - `TERCER ACUERDO (NUEVO NOMBRAMIENTO Y OTORGAMIENTO DE PODERES): Se designa como nuevo ADMINISTRADOR UNICO de la sociedad al ING. ALEJANDRO GÓMEZ RUIZ, a quien se le confiere PODER GENERAL PARA PLEITOS Y COBRANZAS, ACTOS DE ADMINISTRACION Y ACTOS DE DOMINIO para ser ejercido de manera INDIVIDUAL. Se ratifican los poderes vigentes de MARIA ELENA TORRES VEGA.`
- **Expected Deterministic Temporal Resolution for Acta C (`src/pipeline/resolver.py`):**
  1. `numero_escritura` (Constitución original): `"30,100"` (Page 1), `fecha_constitucion`: `"2019-05-10"` (Page 1).
  2. `capital_social_actual`: Resolver compares `2019-05-10` (`$100,000.00`, p. 1) vs `2024-08-20` (`$1,000,000.00`, p. 3). Selects `1000000.0` from Deed `58,400` on Page 3.
  3. `representantes_legales`:
     - Normalizes `"LIC. JUAN CARLOS PÉREZ LÓPEZ"` -> `"JUAN CARLOS PEREZ LOPEZ"`
     - Normalizes `"SR. JUAN C. PEREZ LOPEZ"` -> `"JUAN C PEREZ LOPEZ"`
     - `rapidfuzz.fuzz.token_sort_ratio("JUAN CARLOS PEREZ LOPEZ", "JUAN C PEREZ LOPEZ")` / combined token similarity exceeds `85.0%` (approx `86.4%`).
     - Subtracts Juan Carlos Pérez López from active `representantes_legales` and `poderes`.
     - Final active `representantes_legales` list contains ONLY:
       1. `MARIA ELENA TORRES VEGA` (Apoderada Legal, appointed `2019-05-10`, p. 2)
       2. `ING. ALEJANDRO GÓMEZ RUIZ` (Administrador Único, appointed `2024-08-20`, p. 4)
     - `amendment_history` contains all 4 chronological events (2019 Incorporation, 2024 Capital Increase, 2024 Revocation of Juan Carlos Pérez López with fuzzy score, 2024 Appointment of Alejandro Gómez Ruiz).
