# Data card

## Dataset

- **File:** `data/raw/lung_cancer_combined_22457.csv`
- **Observed dimensions:** 22,457 rows and 17 columns.
- **Observed SHA-256:** `1e0d557990cd7533d2cd37b883fe796b000575d925ed27e33c764f5fe944ee57`
- **Observed prefix:** `1e0d557990cd7533`
- **Contract prefix:** `4ad2333067fcb0a9`
- **Checksum status:** Mismatch. The current file is being analyzed only under the explicit user-approved override. The file's identity relative to the intended project snapshot remains **OPEN**.
- **Access date:** 2026-10-25.

## Source, licence, and collection

- **Original source:** **OPEN**. The supplied repository README contains only a project title and does not identify the dataset source.
- **Licence or reuse permission:** **OPEN**. No licence information was supplied.
- **Collection method, population, and dates:** **OPEN**. These are not described in the available materials.
- **Data acquisition and transformation lineage:** **OPEN**. Handbook task B-4 is implemented in
  `src/lineage.py`; run `python -m src.lineage` after optional Kaggle CSVs are placed in
  `data/external/` (see README there). Latest recorded status lives in
  `docs/lineage_summary.json` and `docs/lineage_report.md`.

## Contents and intended use

The file contains an identifier (`patient_id`), a coded `GENDER` field, `AGE`,
13 symptom/lifestyle flags, and the binary `LUNG_CANCER` recorded label. The
project documentation describes symptom flags as `1 = NO` and `2 = YES`.
`GENDER` code meanings and the meaning or generation process for the recorded
label are **OPEN**. Do not interpret `GENDER` as a verified demographic
definition or the recorded label as a clinical outcome.

The intended use is an educational, integrity-first data science audit of this
file. It is a research prototype, not a clinical tool. No clinical use or
population-level representativeness is established.

## Known issues and limitations

- The observed SHA-256 prefix differs from the project contract prefix.
- Provenance, licence, collection method, population, and label definitions are
  undocumented in the supplied materials.
- `patient_id` is sequential and may encode file order; exclude it from feature
  sets.
- The dataset is a single file; lineage and possible source shifts are not
  independently confirmed.
- The observed file's values describe this file only and do not establish
  representativeness.
- Recorded label `1` has high prevalence in this file. An always-positive rule
  would obtain that same fraction as its accuracy here, despite not using any
  features. Thus accuracy alone is not a meaningful success criterion for this
  imbalanced label distribution.
- The sampling frame and recorded-label meaning are unknown, so this file's
  label prevalence cannot be treated as representative of a real screening
  population. Population-level generalizability remains **OPEN**.

## Unavailable information

The source materials do not provide questionnaire wording, collection dates,
sampling strategy, location, demographic definitions, symptom measurement
protocols, label adjudication details, or participant consent information.
These items remain **OPEN**.
