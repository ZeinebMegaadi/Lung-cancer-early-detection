# Decision log

This log records the handbook decisions and the project’s implementation or
known deviations. The decisions were recorded here on 2026-10-25 at the user's
direction; where the original adoption date or individual decision-maker was
not supplied, it is marked unknown rather than inferred.

| ID | Decision and implementation | Rationale / evidence | Status and revisit condition |
|---|---|---|---|
| D-01 | Treat `data/raw/` as read-only and verify the SHA-256 prefix `4ad2333067fcb0a9` in normal loads. The current file prefix is `1e0d557990cd7533`; the user approved continuing with this snapshot using an explicit `verify=False` override. | Reproducibility and the integrity contract. The expected value is unchanged; see `data/raw/CHECKSUMS.txt` and `docs/results.json`. | **Adopted with approved exception.** Never edit the raw file. Replace the override only after the intended source file and checksum are verified. Original adoption date/individual: unknown. |
| D-02 | Frame the work as an integrity-first descriptive study: ask whether associations in this file persist across documented handling choices, not whether a model is accurate. | The ordered block, duplicates, conflicts, and sensitivity views materially affect interpretation; no population or clinical validation is established. | **Adopted.** Revisit if provenance and the study question are formally changed. Original adoption date/individual: unknown. |
| D-03 | Use view B (exact duplicates removed, keeping the first row) as the primary analysis view; use C and D as pre-specified sensitivities. Show A only to quantify raw-file structure. | This is the minimal, transparent duplicate treatment. **Observed deviation from the handbook's “conclusions are identical in B, C, D” rationale:** H2 and H3 have small nominal findings in B and C, while their effects attenuate in D; this does not support claiming identical test outcomes. Report the observed results, including that H4 has no q-value below 0.05 in D. | **Adopted with discrepancy recorded.** Keep the analysis views; do not suppress or rewrite measured results to match the rationale. Revisit only if an authorized source or instructor specifies a different primary treatment. |
| D-04 | Exclude `patient_id` and row position from all feature sets. Retain ID-order statistics only as explicitly labelled integrity diagnostics. | The ID-only AUC is an audit statistic showing that ordering may carry information; it is not a model result. | **Adopted.** Never add ID or row position to a feature set. |
| D-05 | Do not train a model for this midterm deliverable. Describe possible future modelling only as planned work. | The project scope excludes model training; analyses and figures remain descriptive or integrity-focused. | **Adopted.** Revisit only if the instructor formally extends the scope. |
| D-06 | Refer to `GENDER` as `GENDER_CODE` and retain only its supplied 0/1 codes until a source verifies their meaning. | The mapping is undocumented; assigning sex or gender semantics would be unsupported. | **Adopted.** Revisit if authoritative source documentation becomes available. |
| D-07 | Keep one repository and one Makefile; use seed 42; regenerate figures and numeric results from code. Reports and slides consume generated result/figure outputs rather than manually maintained statistics. | A single reproducible pipeline reduces drift between analysis and deliverables. | **Adopted.** Revisit only if repository or build architecture changes. |
| D-08 | Keep AGE values and flag IQR outliers rather than deleting them. The generated results record 221 IQR flags and 123 rows under age 40. View D is a separate leading-block sensitivity view, not an age-based exclusion rule. | The observed values are plausible under the available documentation and no sentinel-value evidence is established. Separating age handling from row-order handling avoids attributing one policy to another. | **Adopted.** Revisit if verified source documentation shows impossible values or a justified age-specific rule. |
| D-09 | Use “recorded label” and “risk indication”; do not call the target a diagnosis or claim detection. | The label meaning and generation process are unknown, so clinical interpretation is not established. | **Adopted.** Revisit only if provenance confirms the label semantics and wording is approved. |
| D-10 | Keep H1–H5 pre-specified, use alpha 0.05, report effect sizes and confidence intervals, and apply Benjamini–Hochberg correction to the relevant test families. | The large sample can make small effects nominally significant. The report records the family definitions: H1–H3 across views, H4 by feature family per view, and H5 as the raw-file integrity family. | **Adopted.** Revisit only through a documented, pre-analysis change; do not select hypotheses after seeing results. |
| D-11 | The handbook proposes collaborative report authoring in a shared document followed by PDF export. This repository instead maintains generated Markdown and PDF report outputs. | The current project contains a reproducible report source and generated PDF; evidence that a shared document workflow was used was not provided. | **Proposed workflow not evidenced/adopted.** Current output is the generated report. Revisit if the instructor requires shared-document authoring. Original adoption date/individual: not applicable. |
| D-12 | Pin exact package versions in `requirements.txt` and target Python 3.11 or newer. | Pinned dependencies and a documented runtime support reproducibility across machines. | **Adopted.** Revisit if a required package cannot be installed on a supported Python version; update the lock file and validate the full environment. |

## Results discrepancy record

The handbook's D-03 rationale says conclusions are identical in views B, C,
and D. The current generated results do not support that wording for individual
nominal tests. H2 AGE has p-values below 0.05 in B and C but not D; H3
respiratory-count association also has p-values below 0.05 in B and C but not
D. Effect sizes are small in B and C and attenuate toward zero in D. The
feature-family H4 results have no BH-adjusted q-value below 0.05 in D. These
are associations in this file, not population or causal conclusions. The
report's hypothesis table and interpretation retain this distinction; no
results were changed to conform to the handbook's broader headline.
